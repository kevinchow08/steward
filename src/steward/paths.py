"""steward 运行时数据的统一存放位置。

以前这些路径散落在各个模块里，而且大多是 `Path(__file__).parents[N] / "某个目录"`
这种"假设自己正在源码仓库里跑"的写法——一旦 pip 装到 site-packages 下，
这些相对路径全部指向不存在的地方。现在统一收敛到这里，全部是"跟安装位置
无关的绝对路径"：

- 数据库、模型权重、输出报告，都放在 macOS 规范的用户数据目录下
  （~/Library/Application Support/Steward/），跟 index/tag 一直在用的
  数据库路径保持一致，steward 所有运行时数据集中在一处。
- 模型权重体积大（几个 GB）、不随代码版本变，独立分发、放进 MODELS_DIR，
  不打进 wheel（PyPI 有单文件大小限制，而且每次改代码不该让用户重下几 GB）。
"""

from pathlib import Path

APP_DATA_DIR = Path.home() / "Library" / "Application Support" / "Steward"

DEFAULT_DB_PATH = APP_DATA_DIR / "steward.db"
MODELS_DIR = APP_DATA_DIR / "models"
OUTPUT_DIR = APP_DATA_DIR / "output"


def ensure_model_downloaded(local_dir, modelscope_id):
    """确保 local_dir 下有一份可加载的模型；没有就从 ModelScope 下载。

    为什么用 ModelScope 不用 HuggingFace：这个项目主要在国内网络环境下用，
    直连 HF Hub 经常慢/连不上；ModelScope（魔搭，阿里的模型社区）上
    bge-m3 / bge-reranker-v2-m3 都有官方镜像，国内下载快。

    "有没有"的判断标准是 local_dir 下有没有 config.json（模型目录的标志
    文件），不是"目录是不是非空"——避免一个下了一半、内容不完整的目录被
    当成"已经有了"。

    下载进度：snapshot_download 自带 tqdm 进度条，直接打在终端上，这里不用
    另外处理。下载是一次性的，下完缓存在 local_dir，之后不再下。
    """
    local_dir = Path(local_dir)
    if (local_dir / "config.json").exists():
        return

    print(
        f"本地没有 {local_dir.name} 模型，从 ModelScope 下载（一次性，"
        f"几个 GB，下完缓存在本地不再重复下）...",
        flush=True,
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    from modelscope import snapshot_download

    snapshot_download(modelscope_id, local_dir=str(local_dir))
