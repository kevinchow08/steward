"""本地 embedding 模型封装。

这一层只负责把文本转换成向量，不负责保存向量或计算搜索结果。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_MODEL_NAME = "BAAI/bge-m3"
# embeddings.py 位于 src/steward/ 下，向上两层就是项目根目录。
DEFAULT_MODEL_CACHE = Path(__file__).resolve().parents[2] / "models" / "bge-m3"


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
        cache_folder=DEFAULT_MODEL_CACHE,
    ):
        # 延迟导入：只有真正创建 LocalEmbedder 时才加载较重的模型库。
        from sentence_transformers import SentenceTransformer

        model_kwargs = {}
        if device is not None:
            model_kwargs["device"] = device
        if cache_folder is not None:
            cache_folder = Path(cache_folder).expanduser()
            cache_folder.mkdir(parents=True, exist_ok=True)
            model_kwargs["cache_folder"] = str(cache_folder)

        self._model = SentenceTransformer(model_name, **model_kwargs)
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
