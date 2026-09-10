"""开发时的入口：`python main.py <子命令> ...`。

真正的 CLI 逻辑全部在 src/steward/cli.py 里（住在包内部，pip 装完之后靠
入口点 `steward` 命令调用）。这个文件只是为了开发阶段不用先 `pip install`
就能直接跑，做两件事：把 src/ 加进模块搜索路径、然后调 steward.cli.main()。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from steward.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
