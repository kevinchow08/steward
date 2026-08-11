"""基于本地 SQLite 索引执行语义搜索。"""

from dataclasses import dataclass

import numpy as np

from steward.document_index import DEFAULT_DB_PATH, DocumentIndex


@dataclass
class SearchResult:
    """一次搜索命中的文件和对应的最佳文本片段。"""

    path: str
    score: float
    chunk_index: int
    text: str


def search_documents(query, embedder, db_path=DEFAULT_DB_PATH, top_k=5):
    """搜索和用户 query 语义最接近的文件。

    返回 (results, stats) 二元组，包含命中结果列表与详细耗时/计数统计。
    """
    import time

    # 1. 记录 query 生成向量的耗时
    t0 = time.monotonic()
    query_vector = embedder.embed_query(query)
    query_embed_seconds = time.monotonic() - t0

    results_by_document = {}
    chunk_count = 0

    # 2. 记录从数据库读取向量并计算相似度的耗时
    t1 = time.monotonic()
    with DocumentIndex(db_path) as index:
        model_id = index.get_model_id(embedder.info)
        if model_id is None:
            raise ValueError("当前 embedding 模型还没有对应的索引数据")

        for row in index.iter_search_vectors(model_id):
            chunk_count += 1
            chunk_vector = np.frombuffer(row["vector"], dtype=np.float32)
            if chunk_vector.shape[0] != row["dimension"]:
                continue

            score = _cosine_similarity(query_vector, chunk_vector, embedder.info.normalized)
            current = results_by_document.get(row["document_id"])
            if current is None or score > current.score:
                results_by_document[row["document_id"]] = SearchResult(
                    path=row["path"],
                    score=score,
                    chunk_index=row["chunk_index"],
                    text=_compact_text(row["chunk_text"]),
                )

    vector_search_seconds = time.monotonic() - t1

    results = sorted(
        results_by_document.values(),
        key=lambda result: result.score,
        reverse=True,
    )

    stats = {
        "query_embed_seconds": query_embed_seconds,
        "vector_search_seconds": vector_search_seconds,
        "chunk_count": chunk_count,
        "document_count": len(results_by_document),
    }

    return results[:top_k], stats


def _cosine_similarity(left, right, normalized):
    """计算两个向量的相似度；归一化向量可直接点乘，并统一做 [0.0, 1.0] 数值截断防溢出。"""

    if normalized:
        return float(np.clip(np.dot(left, right), 0.0, 1.0))

    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    val = np.dot(left, right) / (left_norm * right_norm)
    return float(np.clip(val, 0.0, 1.0))


def _compact_text(text, max_chars=220):
    """把 chunk 文本压成适合 CLI 展示的一行摘要。"""

    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."
