"""串联扫描、分类、文本提取、分段、embedding 和 SQLite 持久化。"""

import time
from pathlib import Path

from steward import scan
from steward.classifiers import rule_based
from steward.chunking import chunk_tabular_text, chunk_text
from steward.document_index import DEFAULT_DB_PATH, DocumentIndex
from steward.extractors import (
    SUPPORTED_EXTENSIONS,
    _sample_evenly,
    extract_text,
    format_sample_note,
    spread_sample_text,
)
from steward.monitor import ResourceMonitor


BASE_DIR = Path(__file__).resolve().parents[2]
RULES_PATH = BASE_DIR / "config" / "rules.yaml"

# extractors.py 里表格类提取器实际会用到的名字（.py/xlsx/xls）——用来判断
# 一次提取出来的文本要不要走 chunk_tabular_text()（每个 chunk 带表头）而不是
# 走 chunk_text()（普通文章式切分，表格数据切到中间会脱离列名上下文，见
# chunk_tabular_text() 的注释）。不用完整字符串相等判断，是因为 .xls 那边
# 真实混着好几种子格式，提取器名字带着括号里的具体说明（比如
# "openpyxl(伪装成xls的xlsx)"），用 startswith/包含判断能一次覆盖这些变体，
# 不用把每一种具体写法都硬编码进来。
def _is_tabular_extractor(extractor_name):
    if not extractor_name:
        return False
    return (
        extractor_name == "python-csv"
        or extractor_name.startswith("openpyxl")
        or extractor_name.startswith("xlrd")
        or "SpreadsheetML" in extractor_name
    )


# 一份文件切出来的 chunk 数超过这个上限，均匀抽样降到上限以内再去做 embedding——
# 兜底的安全阀，管的是 embedding 成本，对所有格式统一生效，没有例外（早期版本
# 里 JSON 数组在提取阶段单独做过一次"按记录抽样"，实测证明这个专属处理反而让
# 最终覆盖面变差，已经去掉，见 extractors.py 里"代表性抽样工具包"那段注释里的
# 实测数据）。500 这个数字覆盖了一本挺厚的书的量级（500 个 800 字符的 chunk ≈
# 40 万字符），不是精确算出来的，后续发现不合适可以再调。只影响向量检索的
# 覆盖率，不影响打标签——打标签是单独读一段摘要判断，不会因为 chunk 数量多就
# 跟着变慢/变贵，两件事互不影响。
MAX_CHUNKS_PER_DOCUMENT = 500

# extraction.text 本身（不是切出来的 chunk，是完整原始提取文本）超过这个字符数，
# 落库前用 spread_sample_text() 压缩一份代表性摘要再存——管的是数据库体积，
# 跟上面的 chunk 数量上限是两件独立的事：chunk 上限只影响送去 embedding 的
# 那批 chunk，不改 extraction.text 本身；这份原文不管 chunk 有没有被抽样，
# 都会原封不动整份写进 extractions.full_text，真实撞过的例子是一份 118MB 的
# JSON 聊天记录导出文件，不加这道兜底的话会原样存进数据库，tagging 那边读
# full_text 做摘要时也要在这坨超大文本里操作。这一步必须放在切分完成之后——
# chunk_text()/chunk_tabular_text() 要用完整原文切，才能保证切出来的 chunk
# 覆盖全篇（这正是上面 MAX_CHUNKS_PER_DOCUMENT 抽样能做到"均匀覆盖全文"的
# 前提），如果在切分前就先压缩原文，chunk 就只能覆盖被压缩剩下的那一小段，
# 覆盖面反而变差。300 万字符是参考真实语料定的：目前见过最大的两份合法长文档
# （一本书、一份周资料 PDF）原文分别约 56 万、92 万字符，这个阈值留了比它们
# 都宽松的余量，只拦住真正离谱的极端情况，不是精确算出来的。
STORED_TEXT_SAMPLE_THRESHOLD = 3_000_000


def build_index(target_dir, embedder, db_path=DEFAULT_DB_PATH, force=False):
    """为一个目录建立本地 document 内容索引，支持增量（内容没变的文件跳过重新
    处理）和多目录/多盘共享同一个数据库（见 document_index.py 的 DEFAULT_DB_PATH
    注释）。

    embedder 由调用方传入，避免这个函数偷偷加载模型；这样模型加载、
    索引范围和数据库路径都由 CLI 或上层业务明确控制。

    force：默认 False，走增量。传 True 会让下面的"已经是最新"判断直接失效
    （不去查数据库，当成从来没处理过），所有文件不管内容变没变都重新提取/
    分段/向量化一遍。跟 tag 命令的 --force 是同一个用途——每当"处理逻辑本身"
    变了（比如这次新加的 chunks_fts 关键词索引，是靠 save_document_content()
    才会写入的，老数据在增量判断下会一直跳过、永远补不上这个新索引），就需要
    这个开关强制全部重新走一遍，不能指望增量判断自己发现"逻辑变了"。
    """

    # 必须 .resolve()，不能只 .expanduser()——upsert_document() 存文件路径时用的
    # 是 .resolve()（会穿透软链接），如果这里不一致，幽灵清理那步拿去做路径前缀
    # 匹配时会完全对不上、静默失效（真实踩过：macOS 的 /tmp 实际是指向
    # /private/tmp 的软链接，两种写法解析出来的字符串不一样，用没 resolve 过的
    # target_path 做前缀匹配，一个都匹配不到）。resolve() 在路径不存在时不会
    # 报错（strict=False 是默认行为），所以放在下面的存在性检查之前是安全的。
    target_path = Path(target_dir).expanduser().resolve()
    # 目标目录不可达就直接中止，不要往下走——真实风险场景：外置盘这次没连接，
    # 如果不做这个检查，下面"扫描完成后把没扫到的旧记录标记为不存在"那一步
    # 会把"盘暂时不在线"错误地当成"文件全被删除了"，批量抹掉这个盘底下所有
    # 记录。这个检查必须在动手做任何事之前，越早越好。
    if not target_path.exists():
        raise FileNotFoundError(
            f"目标目录不存在或当前不可达（如果是外置盘，检查一下是不是没连接）: {target_path}"
        )

    rules = rule_based.load_rules(RULES_PATH)
    monitor = ResourceMonitor()
    started_at = time.monotonic()
    stats = {
        "scanned_files": 0,
        "indexed_files": 0,
        "success_files": 0,
        "no_text_files": 0,
        "error_files": 0,
        "unsupported_files": 0,
        "skipped_unchanged_files": 0,
        "chunk_count": 0,
        "project_count": 0,
        "removed_files": 0,
    }

    with DocumentIndex(db_path) as index:
        run_id = index.start_run(embedder.info, target_dir=str(target_path))

        # 增量索引：先查出"在当前 embedding 模型下已经有完整向量"的文件，连同
        # 它们当时的 size/mtime 一起拿出来——扫描到某个文件时，如果路径、大小、
        # 修改时间都跟这里记录的完全一致，说明内容没变过，跳过重新提取/分段/
        # 向量化，只刷新它"这次也扫到了"这个状态。换了一个新的 embedding 模型
        # （get_model_id 返回 None）会让这里返回空字典，所有文件都老实走一遍
        # 完整流程，不会因为"以为没变"而漏处理。
        model_id = index.get_model_id(embedder.info)
        up_to_date = {} if force else index.get_up_to_date_documents(model_id)

        # 代码项目根目录（.git/package.json/pyproject.toml 等标记文件命中的目录）
        # 整体登记成一个"项目"单位，不再逐个源码文件走文档分类那条路——项目内部
        # 动辄成千上万个依赖/构建产物文件，拆开当文档处理既不现实也不准确。
        project_roots = scan.find_project_roots(target_path)
        for project_root in project_roots:
            summary = scan.read_project_summary(project_root)
            document_id = index.upsert_document(project_root, "project", run_id=run_id)
            index.save_project_extraction(document_id, summary)
            stats["project_count"] += 1
        if project_roots:
            print(f"  识别到 {len(project_roots)} 个代码项目，已整体登记，跳过其内部文件。", flush=True)

        for file_path in scan.iter_files(target_path, skip_dirs=project_roots):
            stats["scanned_files"] += 1

            # 提取+分段+向量化几百个文件本身需要真实的几分钟，中途没有任何输出的话
            # 很容易被误认为卡死了（之前真实遇到过）。每处理 25 个文件报一次数，
            # 让人能看出它还在正常往前走。
            if stats["scanned_files"] % 25 == 0:
                print(f"  已扫描 {stats['scanned_files']} 个文件...", flush=True)

            resolved_path = str(Path(file_path).expanduser().resolve())
            file_stat = Path(file_path).stat()

            # 每个扫描到的文件都先登记进 documents 表，带上 Week 1 判断出的 basic_type——
            # 这一步以前放在"是否支持提取"判断之后，导致图片/视频/代码/压缩包这些不支持
            # 内容提取的文件压根没有被写进数据库，后续任何环节（打标签、报告）都看不到
            # 它们存在过，是"安静地遗漏"，不是"明确地跳过"。现在两者分开：登记文件是否
            # 存在、属于什么基础类型，谁都不落下；提取/分段/向量化才是只对支持的格式做。
            result = rule_based.classify_file(file_path, rules)
            document_id = index.upsert_document(
                file_path,
                result["basic_type"],
                run_id=run_id,
            )

            # 当前提取器只处理已支持的后缀；其他文件不写入内容索引（没有正文可提取），
            # 但仍计入报告，避免把"没有处理"误认为"处理成功"。
            if Path(file_path).suffix.lower() not in SUPPORTED_EXTENSIONS:
                stats["unsupported_files"] += 1
                monitor.sample()
                continue

            cached = up_to_date.get(resolved_path)
            if cached is not None and cached == (file_stat.st_size, file_stat.st_mtime_ns):
                stats["skipped_unchanged_files"] += 1
                monitor.sample()
                continue

            extraction = extract_text(file_path)

            if _is_tabular_extractor(extraction.extractor):
                chunks = chunk_tabular_text(extraction.text)
            else:
                chunks = chunk_text(extraction.text)

            if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
                original_chunk_count = len(chunks)
                chunks = _sample_evenly(chunks, MAX_CHUNKS_PER_DOCUMENT)
                print(
                    f"  [Warning] {file_path} 切出 {original_chunk_count} 个片段，"
                    f"超过上限 {MAX_CHUNKS_PER_DOCUMENT}，均匀抽样保留 {len(chunks)} "
                    "个参与向量检索（不影响打标签，打标签走的是单独的摘要判断）。",
                    flush=True,
                )

            # 落库前的体积兜底：只改 extraction.text 这一份要存进数据库的副本，
            # 上面已经用完整原文切好的 chunk 不受影响。用 spread_sample_text()
            # 而不是简单截断前面一段——截断只会保留最早的内容，跟直接读开头
            # 是同一个问题（tagging 那边的摘要抽取也是同样的顾虑，两处各自
            # 独立调用同一个函数，互不依赖）。
            if extraction.text and len(extraction.text) > STORED_TEXT_SAMPLE_THRESHOLD:
                original_char_count = len(extraction.text)
                sampled_text = spread_sample_text(
                    extraction.text, STORED_TEXT_SAMPLE_THRESHOLD, pieces=50
                )
                extraction.text = f"{format_sample_note(original_char_count)}\n{sampled_text}"
                print(
                    f"  [Warning] {file_path} 原文 {original_char_count} 字符，"
                    f"超过落库上限 {STORED_TEXT_SAMPLE_THRESHOLD}，均匀抽样压缩后"
                    "落库（不影响上面已经切好的 chunk，chunk 仍覆盖完整原文）。",
                    flush=True,
                )

            texts = [chunk.text for chunk in chunks]
            vectors = embedder.embed_documents(texts)

            index.save_document_content(
                document_id,
                extraction,
                chunks,
                vectors,
                embedder.info,
            )

            stats["indexed_files"] += 1
            stats[f"{extraction.status}_files"] += 1
            stats["chunk_count"] += len(chunks)
            monitor.sample()

        # 幽灵条目清理：target_path 这棵子树下、这次运行没有扫到的旧记录（文件被
        # 删除/改名了），标记为不存在——is_present=0 之后，搜索和标签报告的查询
        # 早就统一在过滤 is_present=1，这一步做完幽灵条目会自动从所有下游消失，
        # 不需要改任何下游查询代码。限定在 target_path 范围内是因为统一数据库下
        # 可能同时存有其它目录/盘的记录，这次运行不能波及它们，见
        # DocumentIndex.mark_missing_as_absent() 的注释。
        removed = index.mark_missing_as_absent(str(target_path), run_id)
        stats["removed_files"] = removed

        stats["elapsed_seconds"] = time.monotonic() - started_at
        stats.update(monitor.stop())
        index.finish_run(run_id, stats)

    return stats
