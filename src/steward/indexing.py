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
    }

    with DocumentIndex(db_path) as index:
        run_id = index.start_run(embedder.info)

        for file_path in scan.iter_files(target_dir):
            stats["scanned_files"] += 1

            # 当前提取器只处理已支持的后缀；其他文件不写入内容索引，
            # 但仍计入报告，避免把“没有处理”误认为“处理成功”。
            if Path(file_path).suffix.lower() not in SUPPORTED_EXTENSIONS:
                stats["unsupported_files"] += 1
                monitor.sample()
                continue

            # 只有确定要进入内容索引的文件，才需要调用 Week 1 分类器，
            # 用它提供 basic_type 元数据。
            result = rule_based.classify_file(file_path, rules)
            document_id = index.upsert_document(
                file_path,
                result["basic_type"],
                run_id=run_id,
            )
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
