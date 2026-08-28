"""遍历目录,产出待分类的文件列表,并识别代码项目根目录。"""

import json
import os
from pathlib import Path


# 只读这几个自描述文件，不递归进代码目录——cowork 分析代码项目时验证过，这是真正
# 有信息量的信号来源（项目名字/README 本身就是自然语言描述），比随机抽几个源码文件
# 有效得多（一个 utils/format.ts 单独拿出来看不出整个项目是干什么的）。
PROJECT_README_NAMES = ["README.md", "readme.md", "README.txt", "CLAUDE.md"]


def read_project_summary(project_root, max_len=3000):
    """读一个项目根目录下的自描述内容，拼成一段文字，供后续分类判断用。"""
    project_root = Path(project_root)
    parts = []

    for name in PROJECT_README_NAMES:
        p = project_root / name
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                parts.append(f"[{name}]\n{text[:max_len]}")
            except OSError:
                pass

    package_json = project_root / "package.json"
    if package_json.is_file():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8", errors="ignore"))
            info = {k: data.get(k) for k in ("name", "description") if data.get(k)}
            if info:
                parts.append(f"[package.json]\n{json.dumps(info, ensure_ascii=False)}")
        except (OSError, json.JSONDecodeError):
            pass

    pyproject = project_root / "pyproject.toml"
    if pyproject.is_file():
        try:
            parts.append(f"[pyproject.toml]\n{pyproject.read_text(encoding='utf-8', errors='ignore')[:1000]}")
        except OSError:
            pass

    return "\n\n".join(parts)


# 出现任意一个,就判定这层目录是一个代码项目的根——这些是各语言/平台生态里近乎
# 通用的项目标记文件,不需要理解内容,是可计算、无歧义的信号，不是猜出来的。
PROJECT_MARKER_NAMES = {
    ".git", "package.json", "pyproject.toml", "setup.py", "requirements.txt",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "oh-package.json5",
}

# 依赖/构建产物目录：内部动辄成千上万个第三方文件，不是用户自己的内容，
# 扫描时直接跳过，不列举内部文件（这是真实撞过的坑：cowork 分析代码目录时，
# 递归列目录直接撞上了工具的 2000 条目录项上限，就是被这类目录撑爆的）。
DEPENDENCY_DIR_NAMES = {
    "node_modules", "venv", "dist", "build", "oh_modules",
    "__pycache__", "target", "Pods", "DerivedData",
}


def find_project_roots(root_dir):
    """从上往下找代码项目根目录：一层目录里出现了项目标记文件，就判定为项目根，
    不再往它内部递归找"更深一层的项目"（比如一个 git 仓库里的子模块，不单独当新项目）。
    没有标记的目录，继续往子目录找，直到找到标记或者走到依赖目录/隐藏目录为止。
    """
    root = Path(root_dir).expanduser().resolve()
    project_roots = []

    def _walk(current: Path):
        try:
            entries = list(os.scandir(current))
        except (PermissionError, FileNotFoundError):
            return

        names = {e.name for e in entries}
        if names & PROJECT_MARKER_NAMES:
            project_roots.append(current)
            return  # 找到项目根就不再往下钻，项目内部不再被当成普通文件/更深的项目

        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            if entry.name.startswith(".") or entry.name in DEPENDENCY_DIR_NAMES:
                continue
            _walk(Path(entry.path))

    _walk(root)
    return project_roots


def iter_files(root_dir, skip_dirs=None):
    """skip_dirs：额外要跳过的目录（绝对路径的集合），用于把已经识别成"项目"的
    子树排除在外——项目内部文件不再走逐文件文档分类这条路，改成整体归为一个项目。
    """
    # Path(...) 把字符串路径包装成 pathlib 的路径对象
    # .expanduser() 把 "~/Downloads" 这种带 ~ 的路径展开成真实的用户目录路径,比如 /Users/kevinchow/Downloads
    root = Path(root_dir).expanduser()
    skip_dirs = {Path(p).resolve() for p in (skip_dirs or [])}

    # os.walk 会自顶向下遍历目录树,每进入一层目录就产出一个三元组:
    # dirpath  —— 当前所在的目录路径
    # dirnames —— 当前目录下的子目录名列表(还没进去)
    # filenames —— 当前目录下的文件名列表
    for dirpath, dirnames, filenames in os.walk(root):
        # 这是 os.walk 的一个常见用法:对 dirnames 做"原地修改"(用 [:] 赋值,而不是 dirnames = ...)
        # os.walk 在产出下一层之前会读取 dirnames 的当前内容来决定往哪些子目录递归,
        # 所以在这里把隐藏目录名过滤掉,os.walk 后面就不会再往 .git、.Trash 这些目录里钻了
        # 如果写成 dirnames = [...](重新绑定成一个新列表),os.walk 拿到的还是旧的引用,过滤就不生效
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
            and d not in DEPENDENCY_DIR_NAMES
            and (Path(dirpath) / d).resolve() not in skip_dirs
        ]

        for filename in filenames:
            # 跳过隐藏文件,比如 .DS_Store
            # 注意:"." 前缀判断隐藏是 macOS/Unix 的命名约定,不是文件系统属性;
            # Windows 的隐藏文件靠元数据里的 Hidden 属性位,跟文件名无关,这里的判断搬到 Windows 上会失效。
            # 现阶段只跑 macOS,先不处理,真要兼容 Windows 时再加 os.stat().st_file_attributes 判断分支。
            if filename.startswith("."):
                continue
            yield Path(dirpath) / filename
