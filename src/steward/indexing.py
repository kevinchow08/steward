"""串联扫描、分类、文本提取、分段、embedding 和 SQLite 持久化。"""

import time
from pathlib import Path

from steward import scan
from steward.classifiers import rule_based
from steward.chunking import chunk_text
from steward.document_index import DEFAULT_DB_PATH, DocumentIndex
from steward.extractors import SUPPORTED_EXTENSIONS, extract_text
from steward.monitor import ResourceMonitor


BASE_DIR = Path(__file__).resolve().parents[2]
RULES_PATH = BASE_DIR / "config" / "rules.yaml"


def build_index(target_dir, embedder, db_path=DEFAULT_DB_PATH):
    """为一个目录建立本地 document 内容索引，支持增量（内容没变的文件跳过重新
    处理）和多目录/多盘共享同一个数据库（见 document_index.py 的 DEFAULT_DB_PATH
    注释）。

    embedder 由调用方传入，避免这个函数偷偷加载模型；这样模型加载、
    索引范围和数据库路径都由 CLI 或上层业务明确控制。
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
        up_to_date = index.get_up_to_date_documents(model_id)

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

            chunks = chunk_text(extraction.text)
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
