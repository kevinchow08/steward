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
    """为一个目录建立本地 document 内容索引。

    embedder 由调用方传入，避免这个函数偷偷加载模型；这样模型加载、
    索引范围和数据库路径都由 CLI 或上层业务明确控制。
    """

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
        "chunk_count": 0,
        "project_count": 0,
    }

    with DocumentIndex(db_path) as index:
        run_id = index.start_run(embedder.info)

        # 代码项目根目录（.git/package.json/pyproject.toml 等标记文件命中的目录）
        # 整体登记成一个"项目"单位，不再逐个源码文件走文档分类那条路——项目内部
        # 动辄成千上万个依赖/构建产物文件，拆开当文档处理既不现实也不准确。
        project_roots = scan.find_project_roots(target_dir)
        for project_root in project_roots:
            summary = scan.read_project_summary(project_root)
            document_id = index.upsert_document(project_root, "project", run_id=run_id)
            index.save_project_extraction(document_id, summary)
            stats["project_count"] += 1
        if project_roots:
            print(f"  识别到 {len(project_roots)} 个代码项目，已整体登记，跳过其内部文件。", flush=True)

        for file_path in scan.iter_files(target_dir, skip_dirs=project_roots):
            stats["scanned_files"] += 1

            # 提取+分段+向量化几百个文件本身需要真实的几分钟，中途没有任何输出的话
            # 很容易被误认为卡死了（之前真实遇到过）。每处理 25 个文件报一次数，
            # 让人能看出它还在正常往前走。
            if stats["scanned_files"] % 25 == 0:
                print(f"  已扫描 {stats['scanned_files']} 个文件...", flush=True)

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

        stats["elapsed_seconds"] = time.monotonic() - started_at
        stats.update(monitor.stop())
        index.finish_run(run_id, stats)

    return stats
