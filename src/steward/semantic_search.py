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

# 稠密检索（向量相似度）的最低相关度门槛——原始余弦相似度低于这个值的候选，
# 不管排第几都不算"真正相关"，直接过滤掉，不进入下面的排名融合——RRF 融合
# 排序只看排名不看原始分数大小，不做这道过滤的话，一个语料里根本不存在的
# 话题也会被强行凑出 top_k 个"看起来正常"的结果，融合分数的数值量级跟真正
# 相关的查询没有可辨识的区别，用户没法从结果本身看出"这次是真的搜到了"还是
# "纯粹是矬子里拔将军"。关键词检索（FTS5 MATCH）不需要类似的门槛：只要命中
# 了，说明查询词真的逐字出现在文本里，本身就是有意义的信号，不存在"匹配到了
# 但其实不相关"这种模糊地带。
#
# 0.5 这个数字目前只有 4 个真实查询词的证据支撑（对着一个语料里确实不存在
# 的话题"露营装备推荐"查询，最高相似度只有 0.485；对着 3 个真实存在的话题
# 查询，最低的有 0.6+），样本量不够下"这是稳定分界"的结论，只能算"当前证据
# 支撑的一个估计值"。**这个数字是跟具体 embedding 模型 + 具体语料的内容分布
# 绑定的，不是一个通用常数**——换一个 embedding 模型（不同模型的向量空间
# 几何形状不同，"多相似算相似"这把尺子会整体平移）、或者语料话题范围发生
# 大幅变化，都需要重新用真实查询词测一遍再校准，不能假设 0.5 能直接照搬。
# 同一批文件、同一个模型重新建索引不受影响——embedding 是确定性计算，
# 同样的文本喂给同样的模型算出来的向量是一样的。
#
# 曾经试过一个"不用绝对数值，看这次查询的分数比全库均值高几个标准差"的
# 动态方案，指望它能自动适应模型/语料变化——实测下来效果不如直接卡绝对值：
# "露营装备推荐"最高分离均值 4.29 个标准差，"工程师思维"是 4.46 个标准差，
# 两者差距很小，没有绝对分数（0.485 vs 0.6+）分得开，所以没有采用，还是用
# 更简单、目前证据更支持的绝对阈值，但要如实承认它的适用范围窄。
_MIN_DENSE_SCORE = 0.5


def search_documents(query, embedder, db_path=DEFAULT_DB_PATH, top_k=5):
    """混合检索：向量语义相似度 + 关键词精确匹配（FTS5 + bm25）+ 标签关键词
    匹配（同样是 FTS5 + bm25，但匹配的是文档整体的标签，不是某个 chunk），
    三路结果按排名融合（Reciprocal Rank Fusion），不是按原始分数加权——三路
    分数的数值尺度完全不是一回事（余弦相似度是 0~1 的有界值，bm25 是跟语料
    规模有关的无界值），硬要按百分比加权需要手调系数，而且没有验证数据支撑，
    不可靠。RRF 只看各自的排名，不比较原始分数，天然公平。

    标签这一路存在的意义：向量/关键词检索都是按 chunk 粒度算分的，一份文档
    整体的主题如果没有集中体现在某一个 800 字符的 chunk 里，会在那两路检索
    里"隐形"——但这个主题恰恰是打标签时模型看完整篇之后才提炼出来的，标签
    检索能补上这个盲区（真实案例：一份带着"工程师思维"这个标签的文档，
    在向量+关键词两路混合检索里排到了第 2 名，第 1 名是一份没有这个标签、
    只是话题沾边的文档）。

    关键词/标签这两路都是字符三元组匹配（trigram），不是真正的分词，查询词
    至少要 3 个字符才能命中——2 字短查询这两路都会搜不到东西，这是已知的
    设计代价，不是 bug，向量检索没有长度限制，能接住这类短查询。

    向量检索这一路还有一个最低相关度门槛（见 _MIN_DENSE_SCORE），原始相似度
    不够的候选不会进入这一路的排名——所以查询一个语料里根本不存在的话题，
    可能三路都是空结果，`document_count` 会是 0 或很小的数字，不会被强行
    凑出 top_k 个看起来正常、实际不相关的结果。

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

        t3 = time.monotonic()
        tag_ranking = _tag_search(index, query)
        tag_seconds = time.monotonic() - t3

    merged = _rrf_merge(dense_ranking, sparse_ranking, tag_ranking)
    results = merged[:top_k]

    stats = {
        "query_embed_seconds": query_embed_seconds,
        "vector_search_seconds": dense_seconds,
        "keyword_search_seconds": sparse_seconds,
        "tag_search_seconds": tag_seconds,
        "chunk_count": chunk_count,
        "document_count": len(merged),
        "dense_hit_count": len(dense_ranking),
        "sparse_hit_count": len(sparse_ranking),
        "tag_hit_count": len(tag_ranking),
    }

    return results, stats


def _dense_search(index, model_id, query_vector, normalized):
    """向量检索：把全部 chunk 向量一次性拼成一个矩阵，跟 query 向量做一次批量
    矩阵乘法算余弦相似度，不是之前那种逐行 Python 循环手算点积再比较——语料
    一大，矩阵运算比 Python 循环快得多。

    同一个文档命中好几个 chunk 时只保留分数最高的那个；最高分低于
    _MIN_DENSE_SCORE 的文档直接不进入返回结果，不管它在全库里排第几——
    这一步是本函数的返回值语义变化：以前"返回全部命中"，现在"只返回真正
    相关的命中"，返回的列表可能比全库文档数少得多，甚至是空列表（说明这次
    查询在向量语义这一路上没找到真正相关的内容）。
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
    filtered = [
        (doc_id, result) for doc_id, (score, result) in ranking if score >= _MIN_DENSE_SCORE
    ]
    return filtered, len(rows)


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


def _tag_search(index, query):
    """标签关键词检索：一份文档只有一行索引记录（不像 chunk 那样一份文档
    可能命中好几行），不需要像 _sparse_search() 那样按文档聚合去重。
    展示用的"文本"是命中的标签原文，不是某一段正文——这份结果的价值本来
    就是"这份文档整体被打上了什么标签"，不是"文档里某一句话写了什么"。
    """
    rows = index.search_document_tags(query)
    ranking = []
    for row in rows:
        ranking.append((
            row["document_id"],
            SearchResult(
                path=row["path"],
                score=row["bm25_score"],
                chunk_index=-1,
                text=f"[标签命中] {row['tags_text']}",
            ),
        ))
    return ranking


def _rrf_merge(*rankings):
    """把任意多路排名（每一路都是排好序的 (doc_id, SearchResult) 列表）融合成
    一份排名，不限定两路——三路（向量+关键词+标签）、以后想加第四路，直接
    多传一个参数进来就行，不需要再新增一段几乎一样的循环。

    scores 存的是"名次贡献值的累加"，不是原始的向量相似度或 bm25 分数——各路
    的原始分数单位、方向都不一样（向量是 0~1 越大越好，bm25 无界越小越好），
    从来没有被直接相加过。真正参与相加的，是每一路各自已经排好序之后的
    "排第几名"（rank，从 0 开始），用同一个公式 1/(_RRF_K + rank + 1) 转成
    "贡献值"——排名越靠前贡献值越大，这个贡献值才是各路统一的"货币"。
    scores.get(doc_id, 0.0) 拿的是这个文档在别的路里可能已经累积下来的贡献值
    （没有就是 0.0），所以同一个文档如果好几路都命中，会在这里被自然地
    "叠加"起来，不需要额外判断。

    展示用的片段按 rankings 参数的传入顺序决定优先级——排在前面的那一路
    如果命中了这份文档，就用它的片段展示；后面几路的 setdefault 只在前面
    都没命中这份文档时才会生效。调用方按"最想展示哪路的片段"来决定传参顺序
    （目前是向量 > 关键词 > 标签，标签命中给的是标签原文，不是正文片段，
    优先级放最后）。
    """
    scores = {}
    display_result = {}

    for ranking in rankings:
        for rank, (doc_id, result) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
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
