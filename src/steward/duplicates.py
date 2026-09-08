"""精确重复文件检测——只判断"内容 byte 级完全一样"，不做近似相似判断，
不做任何删除/移动，纯只读扫描 + 报告。

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
