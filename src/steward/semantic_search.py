"""基于本地 SQLite 索引执行语义搜索——向量检索 + 关键词检索的混合排序。"""

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


# Reciprocal Rank Fusion 用到的经验常数，是这个算法在信息检索领域惯用的固定值，
# 不是需要针对我们语料去调的参数——RRF 的设计目的就是不依赖调参也能给出合理的
# 融合结果，见 _rrf_merge() 的说明。
_RRF_K = 60


def search_documents(query, embedder, db_path=DEFAULT_DB_PATH, top_k=5):
    """混合检索：向量语义相似度 + 关键词精确匹配（FTS5 + bm25），两路结果按
    排名融合（Reciprocal Rank Fusion），不是按原始分数加权——两路分数的数值
    尺度完全不是一回事（余弦相似度是 0~1 的有界值，bm25 是跟语料规模有关的
    无界值），硬要按百分比加权需要手调一个系数，而且这个系数没有验证数据支撑，
    不可靠。RRF 只看两路各自的排名，不比较原始分数，天然公平。

    关键词检索用的是字符三元组匹配（trigram），不是真正的分词，查询词至少要
    3 个字符才能命中——2 字短查询这一路会搜不到东西，这是已知的设计代价，
    不是 bug，向量检索这一路没有长度限制，能接住这类短查询，两路互补正是
    "混合"检索存在的意义。

    返回 (results, stats) 二元组，包含命中结果列表与详细耗时/计数统计。
    """
    import time

    t0 = time.monotonic()
    query_vector = embedder.embed_query(query)
    query_embed_seconds = time.monotonic() - t0

    with DocumentIndex(db_path) as index:
        model_id = index.get_model_id(embedder.info)
        if model_id is None:
            raise ValueError("当前 embedding 模型还没有对应的索引数据")

        t1 = time.monotonic()
        dense_ranking, chunk_count = _dense_search(index, model_id, query_vector, embedder.info.normalized)
        dense_seconds = time.monotonic() - t1

        t2 = time.monotonic()
        sparse_ranking = _sparse_search(index, query)
        sparse_seconds = time.monotonic() - t2

    merged = _rrf_merge(dense_ranking, sparse_ranking)
    results = merged[:top_k]

    stats = {
        "query_embed_seconds": query_embed_seconds,
        "vector_search_seconds": dense_seconds,
        "keyword_search_seconds": sparse_seconds,
        "chunk_count": chunk_count,
        "document_count": len(merged),
        "dense_hit_count": len(dense_ranking),
        "sparse_hit_count": len(sparse_ranking),
    }

    return results, stats


def _dense_search(index, model_id, query_vector, normalized):
    """向量检索：把全部 chunk 向量一次性拼成一个矩阵，跟 query 向量做一次批量
    矩阵乘法算余弦相似度，不是之前那种逐行 Python 循环手算点积再比较——语料
    一大，矩阵运算比 Python 循环快得多。

    同一个文档命中好几个 chunk 时只保留分数最高的那个，返回按分数从高到低
    排好序的 (document_id, SearchResult) 列表。
    """
    rows = list(index.iter_search_vectors(model_id))
    if not rows:
        return [], 0

    # (N, dimension) 的矩阵，N 是全部 chunk 数——每一行是一个 chunk 的向量。
    vectors = np.array(
        [np.frombuffer(row["vector"], dtype=np.float32) for row in rows],
        dtype=np.float32,
    )
    query_vec = np.asarray(query_vector, dtype=np.float32)

    if normalized:
        # 向量都是单位向量时，点积本身就是余弦相似度，vectors @ query_vec
        # 一次矩阵乘法就是全部 N 个 chunk 各自跟 query 的点积。
        scores = np.clip(vectors @ query_vec, 0.0, 1.0)
    else:
        vec_norms = np.linalg.norm(vectors, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = vec_norms * query_norm
        denom = np.where(denom == 0, 1.0, denom)  # 避免除以 0
        scores = np.clip((vectors @ query_vec) / denom, 0.0, 1.0)

    best_by_document = {}
    for row, score in zip(rows, scores.tolist()):
        doc_id = row["document_id"]
        current = best_by_document.get(doc_id)
        if current is None or score > current[0]:
            best_by_document[doc_id] = (
                score,
                SearchResult(
                    path=row["path"],
                    score=score,
                    chunk_index=row["chunk_index"],
                    text=_compact_text(row["chunk_text"]),
                ),
            )

    ranking = sorted(best_by_document.items(), key=lambda item: item[1][0], reverse=True)
    return [(doc_id, result) for doc_id, (_, result) in ranking], len(rows)


def _sparse_search(index, query):
    """关键词检索：按文档聚合（同一文档命中多个 chunk，只保留 bm25 分数最好的
    那个），返回按 bm25 从好到坏排好序的 (document_id, SearchResult) 列表。
    bm25 越小代表匹配度越高，这里"更好"的判断是 <，跟向量那边的 > 刚好相反，
    别看错方向——真实测过验证过这个符号约定，不是凭直觉猜的。
    """
    rows = index.search_keyword_chunks(query)
    best_by_document = {}
    for row in rows:
        doc_id = row["document_id"]
        current = best_by_document.get(doc_id)
        if current is None or row["bm25_score"] < current[0]:
            best_by_document[doc_id] = (
                row["bm25_score"],
                SearchResult(
                    path=row["path"],
                    score=row["bm25_score"],
                    chunk_index=row["chunk_index"],
                    text=_compact_text(row["chunk_text"]),
                ),
            )

    ranking = sorted(best_by_document.items(), key=lambda item: item[1][0])
    return [(doc_id, result) for doc_id, (_, result) in ranking]


def _rrf_merge(dense_ranking, sparse_ranking):
    # scores 存的是"名次贡献值的累加"，不是原始的向量相似度或 bm25 分数——两边
    # 的原始分数单位、方向都不一样（向量是 0~1 越大越好，bm25 无界越小越好），
    # 从来没有被直接相加过。真正参与相加的，是 dense_ranking/sparse_ranking
    # 各自已经排好序之后的"排第几名"（rank，从 0 开始），用同一个公式
    # 1/(_RRF_K + rank + 1) 转成"贡献值"——排名越靠前贡献值越大，这个贡献值
    # 才是两边统一的"货币"。scores.get(doc_id, 0.0) 拿的是这个文档在另一路
    # 循环里可能已经累积下来的贡献值（没有就是 0.0），所以同一个文档如果两路
    # 都命中，会在这里被自然地"叠加"起来，不需要额外判断。
    scores = {}
    display_result = {}

    for rank, (doc_id, result) in enumerate(dense_ranking):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        display_result.setdefault(doc_id, result)

    for rank, (doc_id, result) in enumerate(sparse_ranking):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        # 展示用的片段优先用向量检索那边命中的（上面那个循环先跑，setdefault
        # 只在向量检索没命中这份文档时才会被这里的关键词命中片段填上）。
        display_result.setdefault(doc_id, result)

    ranked_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
    merged = []
    for doc_id in ranked_ids:
        result = display_result[doc_id]
        merged.append(SearchResult(
            path=result.path,
            score=scores[doc_id],
            chunk_index=result.chunk_index,
            text=result.text,
        ))
    return merged


def _compact_text(text, max_chars=220):
    """把 chunk 文本压成适合 CLI 展示的一行摘要。"""

    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."
