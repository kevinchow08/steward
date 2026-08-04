"""串联扫描已提取文本的文档，调用分类与打标签引擎，并批量持久化至 SQLite。"""

import time
from pathlib import Path
from typing import Dict, Any

from steward.document_index import DEFAULT_DB_PATH, DocumentIndex
from steward.semantic_classifier import classify_document_text


def build_classifications(db_path=DEFAULT_DB_PATH, confidence_threshold=0.70) -> Dict[str, Any]:
    """为 SQLite 中所有已提取文本的文档生成主分类与标签并持久化。"""

    started_at = time.monotonic()
    stats = {
        "total_documents": 0,
        "classified_count": 0,
        "unclassified_count": 0,
        "tag_bindings_count": 0,
        "elapsed_seconds": 0.0,
    }

    with DocumentIndex(db_path) as index:
        # 从数据库中读取所有状态为 success 且包含 full_text 的提取记录
        cursor = index.connection.execute(
            """
            SELECT
                d.id AS document_id,
                d.path AS path,
                d.basic_type AS basic_type,
                x.full_text AS full_text
            FROM documents d
            JOIN extractions x ON x.document_id = d.id
            WHERE d.is_present = 1
              AND x.status = 'success'
            """
        )
        rows = cursor.fetchall()
        stats["total_documents"] = len(rows)

        for row in rows:
            doc_id = row["document_id"]
            file_path = row["path"]
            basic_type = row["basic_type"]
            full_text = row["full_text"]

            # 1. 调用分类引擎进行文本推理
            res = classify_document_text(
                text=full_text,
                file_path=file_path,
                basic_type=basic_type,
            )

            # 2. 持久化到 SQLite
            index.save_classification(
                document_id=doc_id,
                category_name=res.category,
                tags=res.tags,
                confidence=res.confidence,
                status=res.status,
                reasoning=res.reasoning,
            )

            # 3. 统计计数
            if res.status == "classified":
                stats["classified_count"] += 1
            else:
                stats["unclassified_count"] += 1

            stats["tag_bindings_count"] += len(res.tags)

        stats["elapsed_seconds"] = round(time.monotonic() - started_at, 3)

    return stats
