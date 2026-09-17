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


def _group_by_content(candidate_paths):
    """核心分组算法：给一批候选路径，返回内容完全相同的文件分组。

    不关心这批路径是怎么来的（扫文件系统 or 查数据库）——find_duplicates()
    和 find_duplicates_from_index() 都是"拿到一批候选路径"之后调这个函数，
    这段"怎么判断重复"的逻辑只写一份，不重复。

    两阶段过滤，只有真正需要才去读文件内容：
    1. 先按文件大小分组（stat() 现成的信息，不用读文件内容）——大小在
       候选集里独一无二的文件，不可能跟别的文件内容相同，直接排除，不用
       浪费时间去读它的字节算哈希。
    2. 只对"大小有重复"的那一批文件真正计算哈希，再按哈希分组，分组里
       文件数 > 1 的才是真正的重复。

    判断依据只看文件内容本身——文件名、所在目录都不参与判断（真实语料里
    见过好几个文件夹各自都有一份叫"餐饮1.pdf"的发票，但金额、日期完全
    不同，同名不代表内容一样，这个坑必须靠只认内容哈希来避开）。

    返回一个列表，每个元素是 (sha256, [文件路径列表], 单份大小, 浪费字节数)，
    按这组文件加起来浪费的空间从大到小排序。
    """
    # 第一阶段：按大小分组
    by_size = defaultdict(list)
    for file_path in candidate_paths:
        try:
            size = Path(file_path).stat().st_size
        except (FileNotFoundError, PermissionError):
            # 候选路径读不到了（权限问题、扫描/建索引之后文件被删了），
            # 跳过，不是这个功能要处理的问题。
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


def find_duplicates(target_dir):
    """扫描 target_dir，返回内容完全相同的文件分组。

    每次都是一次独立的全新文件系统扫描，不碰数据库、不需要先跑过
    index——随时能用，代价是每次只能看到这一个目录，看不到"同一份内容
    还存在别的、这次没扫到的目录里"这种跨目录重复。要覆盖这种情况，用
    下面的 find_duplicates_from_index()。

    跟 index 命令一样复用 scan.find_project_roots()/scan.iter_files()
    跳过代码项目内部——一个 node_modules 目录里几千个文件互相之间大量
    雷同是正常现象，不是这次要抓的"用户自己的重复文件"，硬扫进来只会让
    报告充满噪声。
    """
    target_path = Path(target_dir).expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"目标目录不存在或当前不可达: {target_path}")

    project_roots = scan.find_project_roots(target_path)
    candidate_paths = scan.iter_files(target_path, skip_dirs=project_roots)
    return _group_by_content(candidate_paths)


def find_duplicates_from_index(db_path, target_dir=None):
    """不重新扫文件系统，改用数据库里已经登记过的文件当候选集，找重复。

    候选来源是 document_index.DocumentIndex.list_present_paths()——只看
    is_present=1 的文件（已经被标记为消失/改名的旧记录不参与），数据库
    本身是可以跨目录、跨盘共享的（见 index_runs.target_dir、
    documents.volume_label），所以这个模式天然能发现"同一份内容分别存在
    两个不同的、各自都建过索引的目录里"这种 find_duplicates() 单目录扫描
    发现不了的跨目录重复。

    target_dir=None：候选集是数据库里全部已索引、当前仍存在的文件（真正
    的"跨全库查重"）。传了 target_dir：候选集限定在这棵子树下，行为更
    接近 find_duplicates()，但省一次重新扫描文件系统的开销，且能排除
    已经消失的旧记录。

    这个模式的局限：只能看到"已经跑过 index 的文件"——没建过索引的目录，
    这个模式看不到，需要用 find_duplicates() 直接扫。两个模式互补，不是
    互相替代。
    """
    from steward.document_index import DocumentIndex

    with DocumentIndex(db_path=db_path) as index:
        candidate_paths = index.list_present_paths(target_dir=target_dir)

    return _group_by_content(candidate_paths)


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


def _candidate_trash_dirs(original_path):
    """给一个文件原来所在的路径，返回它被删除时可能落进的废纸篓目录。

    macOS 的废纸篓不是只有一个：本机磁盘统一用 ~/.Trash；外置盘各自有
    独立的废纸篓，在 /Volumes/<盘名>/.Trashes/<当前用户 UID>/ 下——判断
    逻辑跟 document_index._derive_volume_label() 判断"这个文件躺在哪块
    盘上"是同一个思路（看路径是不是以 /Volumes/ 开头），只是这里要的是
    实际的废纸篓目录路径，不是拿来展示的盘名字符串，所以单独写，不直接
    复用那个函数。
    """
    import os

    resolved = Path(original_path).resolve()
    parts = resolved.parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        volume_root = Path(*parts[:3])
        return [volume_root / ".Trashes" / str(os.getuid())]
    return [Path.home() / ".Trash"]


def find_in_trash(sha256, size_bytes, original_path):
    """在废纸篓里找一份内容（SHA256 + 大小）匹配的文件，返回它当前在
    废纸篓里的路径；没找到返回 None。

    不靠文件名匹配——文件移入废纸篓时，如果废纸篓里已经有同名文件，
    macOS 会自动改名（加数字后缀），文件名不可靠；SHA256 和 size_bytes
    是从删除日志（duplicates_cleanup_log.jsonl）里原样带过来的，删除
    那一刻就记好了，是可靠的判断依据，不用重新猜。先比 size_bytes 再算
    哈希，避免对废纸篓里每一个文件都算一遍哈希——跟 _group_by_content()
    "先按大小分组、只对真正可能重复的文件算哈希"是同一个性能优化思路。

    如果这份文件已经被还原过、或者废纸篓被清空过，这里自然会返回
    None——调用方不需要额外判断"是不是已经还原过"，找不到就是找不到，
    统一处理。
    """
    for trash_dir in _candidate_trash_dirs(original_path):
        if not trash_dir.exists():
            continue
        for candidate in trash_dir.iterdir():
            if not candidate.is_file():
                continue
            try:
                if candidate.stat().st_size != size_bytes:
                    continue
                if _hash_file(candidate) == sha256:
                    return candidate
            except (FileNotFoundError, PermissionError):
                continue
    return None


def restore_from_trash(trash_path, original_path):
    """把废纸篓里的文件挪回原来的路径。

    如果原路径现在已经有文件了（用户后来自己在那个位置又新建/放了一份
    同名文件），拒绝覆盖，抛出异常交给调用方处理——这是行动层操作，宁可
    保守报错，不做任何可能悄悄覆盖用户新数据的隐式操作。
    """
    import shutil

    original_path = Path(original_path)
    if original_path.exists():
        raise FileExistsError(f"原路径已经有文件了，不覆盖: {original_path}")
    original_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(trash_path), str(original_path))
