"""本地 embedding 模型封装。

这一层只负责把文本转换成向量，不负责保存向量或计算搜索结果。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from steward.paths import MODELS_DIR, ensure_model_downloaded


DEFAULT_MODEL_NAME = "BAAI/bge-m3"
# 模型权重放在统一的用户级目录下（见 steward/paths.py），不打进 pip 包——
# 体积几个 GB，独立分发/首次运行自动下载，代码去这个约定位置找。
DEFAULT_MODEL_DIR = MODELS_DIR / "bge-m3"
# ModelScope 上的仓库 ID，本地没有模型时从这里下。
MODELSCOPE_ID = "BAAI/bge-m3"


@dataclass
class EmbeddingModelInfo:
    """描述生成向量所使用的模型，写入索引时用于追踪版本。"""

    model_name: str
    dimension: int
    normalized: bool


class LocalEmbedder:
    """使用 sentence-transformers 在本机生成文本向量。"""

    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        device=None,
        normalize_embeddings=True,
        model_dir=DEFAULT_MODEL_DIR,
        modelscope_id=MODELSCOPE_ID,
    ):
        # 延迟导入：只有真正创建 LocalEmbedder 时才加载较重的模型库。
        from sentence_transformers import SentenceTransformer

        model_dir = Path(model_dir).expanduser()
        # 本地没有就从 ModelScope 下载（一次性，带进度条），下完缓存在 model_dir。
        ensure_model_downloaded(model_dir, modelscope_id)

        model_kwargs = {}
        if device is not None:
            model_kwargs["device"] = device

        # 直接从本地目录加载——model_dir 是一个平铺的模型文件目录
        # （config.json / pytorch_model.bin / tokenizer 等直接在里面），
        # 不是 HF hub 的缓存布局。ensure_model_downloaded 保证了它存在。
        self._model = SentenceTransformer(str(model_dir), **model_kwargs)

        self._model_name = model_name
        self._normalize_embeddings = normalize_embeddings

        # 新版 sentence-transformers 使用 get_embedding_dimension。
        # 旧版本仍可能只有旧名字，因此保留一个明确的兼容分支。
        if hasattr(self._model, "get_embedding_dimension"):
            dimension = self._model.get_embedding_dimension()
        else:
            dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError("无法从 embedding 模型读取向量维度")

        self.info = EmbeddingModelInfo(
            model_name=model_name,
            dimension=int(dimension),
            normalized=normalize_embeddings,
        )

    def embed_documents(self, texts, batch_size=16):
        """批量生成文档片段向量，返回 shape 为 (数量, 维度) 的数组。"""

        return self._encode(texts, encode_kind="document", batch_size=batch_size)

    def embed_query(self, text):
        """生成一条用户查询的向量，返回 shape 为 (1, 维度) 的数组。"""

        vectors = self._encode([text], encode_kind="query", batch_size=1)
        return vectors[0]

    def _encode(self, texts, encode_kind, batch_size):
        if not texts:
            return np.empty((0, self.info.dimension), dtype=np.float32)

        # 新版 sentence-transformers 为非对称搜索提供 encode_document/query。
        # 这里显式选择方法，不把逻辑压缩进 getattr，方便阅读和调试。
        if encode_kind == "document" and hasattr(self._model, "encode_document"):
            encode_method = self._model.encode_document
        elif encode_kind == "query" and hasattr(self._model, "encode_query"):
            encode_method = self._model.encode_query
        else:
            encode_method = self._model.encode
        vectors = encode_method(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        )
        return np.asarray(vectors, dtype=np.float32)
