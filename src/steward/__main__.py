"""让 `python -m steward <子命令> ...` 也能用，等价于装完之后的 `steward` 命令。"""

from steward.cli import main

if __name__ == "__main__":
    main()
