"""精确重复文件检测——只判断"内容 byte 级完全一样"，不做近似相似判断。

`find_duplicates()` 本身还是纯只读扫描 + 报告，不碰任何文件。这个模块
另外提供两个给"清理"场景用的小工具：`pick_keeper()`（从一组重复文件里
选出默认建议保留哪一份，纯计算，不碰文件系统）和 `move_to_trash()`
（真正会改动文件系统的唯一入口，移入系统废纸篓、可撤销，不是永久删除）。
真正的交互式清理流程（逐组确认、写操作日志）在 main.py 的
`run_duplicates_clean()` 里，这里只放"选谁""怎么删"这两个可以独立测试
的判断/动作，不掺业务流程。

跟 index/tag 这条链路完全独立：不读写数据库，每次调用都是一次全新扫描，
不做增量、不做持久化——这是刻意的最小切片，先把"从零到一"跑通，增量/
持久化留给以后真的需要的时候再加。
"""

import hashlib
from collections import defaultdict
from pathlib import Path

from steward import scan


# 算哈希时每次从文件里读多少字节，不是一次性把整个文件读进内存——大文件
# （这个项目语料里见过几十上百 MB 的 PDF/JSON）一次性读入会占用对应大小的
# 内存，流式读取只需要固定这么大的缓冲区，跟文件多大无关。1MB 是常见的
# 折中选择，不是精确调出来的数字。
_HASH_CHUNK_SIZE = 1024 * 1024


def _hash_file(path):
    """流式计算一个文件的 SHA256——两个文件哈希一样，内容就是 100% 一样
    （哈希碰撞的概率低到可以忽略，这是数学性质，不是"大概率对"）。"""

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def find_duplicates(target_dir):
    """扫描 target_dir，返回内容完全相同的文件分组。

    两阶段过滤，只有真正需要才去读文件内容：
    1. 先按文件大小分组（stat() 现成的信息，不用读文件内容）——大小在
       全目录里独一无二的文件，不可能跟别的文件内容相同，直接排除，不用
       浪费时间去读它的字节算哈希。
    2. 只对"大小有重复"的那一批文件真正计算哈希，再按哈希分组，分组里
       文件数 > 1 的才是真正的重复。

    判断依据只看文件内容本身——文件名、所在目录都不参与判断（真实语料里
    见过好几个文件夹各自都有一份叫"餐饮1.pdf"的发票，但金额、日期完全
    不同，同名不代表内容一样，这个坑必须靠只认内容哈希来避开）。

    跟 index 命令一样复用 scan.find_project_roots()/scan.iter_files()
    跳过代码项目内部——一个 node_modules 目录里几千个文件互相之间大量
    雷同是正常现象，不是这次要抓的"用户自己的重复文件"，硬扫进来只会让
    报告充满噪声。

    返回一个列表，每个元素是 (sha256, [文件路径列表])，按这组文件加起来
    浪费的空间从大到小排序。
    """
    target_path = Path(target_dir).expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"目标目录不存在或当前不可达: {target_path}")

    project_roots = scan.find_project_roots(target_path)

    # 第一阶段：按大小分组
    by_size = defaultdict(list)
    for file_path in scan.iter_files(target_path, skip_dirs=project_roots):
        try:
            size = Path(file_path).stat().st_size
        except (FileNotFoundError, PermissionError):
            # 扫描到但读不到的文件（权限问题、扫描过程中被删了），跳过，
            # 不是这个功能要处理的问题。
            continue
        by_size[size].append(file_path)

    # 第二阶段：只对大小有重复的文件算哈希，再按哈希分组
    by_hash = defaultdict(list)
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        for path in paths:
            try:
                digest = _hash_file(path)
            except (FileNotFoundError, PermissionError):
                continue
            by_hash[digest].append(path)

    groups = []
    for digest, paths in by_hash.items():
        if len(paths) < 2:
            continue
        size = Path(paths[0]).stat().st_size
        wasted_bytes = size * (len(paths) - 1)
        groups.append((digest, paths, size, wasted_bytes))

    groups.sort(key=lambda g: g[3], reverse=True)
    return groups


def pick_keeper(paths):
    """从一组重复文件里，选出一份默认建议保留的，其余是默认建议删除的。

    规则：修改时间(mtime)最早的那份——同样内容，越早出现的越接近"原始
    文件"，后面的更像是复制/搬运出来的副本。mtime 相同时（比如同一次
    操作批量拷贝出来的），再比路径长度，选最短的（启发式：路径越短，
    往往越接近"主目录"，嵌套很深的路径更像是归档/备份出来的副本）。

    这只是一个默认建议，不是强制结论——调用方会把分组里每个文件的路径
    和 mtime 完整展示给用户，用户可以在这个默认选择之上手动改选保留哪份。

    返回 (保留路径, 建议删除的路径列表)。
    """

    def sort_key(path):
        try:
            mtime = Path(path).stat().st_mtime
        except (FileNotFoundError, PermissionError):
            # 读不到 mtime 的文件（比如扫描之后被删了）不该被选成"保留项"，
            # 排到最后面去。
            mtime = float("inf")
        # scan.iter_files() 产出的是 Path 对象，不是字符串，len() 直接作用
        # 在 Path 上会报错（Path 没有 __len__），要转成字符串再比长度。
        return (mtime, len(str(path)))

    # min() 对每个 path 都调一次 sort_key()，比较得到的 (mtime, 路径长度)
    # 二元组——tuple 比较是逐位比较，先比 mtime，相同再比长度——返回"最小"
    # 的那个 key 所对应的原始 path（不是返回 key 本身）。
    keeper = min(paths, key=sort_key)
    to_delete = [p for p in paths if p != keeper]
    return keeper, to_delete


def move_to_trash(path):
    """把一个文件移入系统废纸篓——可以从废纸篓手动还原，不是永久删除。

    这是这个模块里唯一一个真正会改动文件系统的函数，独立出来方便单独
    测试/替换，不要在别的地方直接调 os.remove() 之类的永久删除操作。
    """
    from send2trash import send2trash

    send2trash(str(path))
