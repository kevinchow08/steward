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
