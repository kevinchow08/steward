"""基于本地 SQLite 索引执行语义搜索——向量+关键词+标签三路粗筛，
再用 cross-encoder 重排序做最终精排（先粗筛后精排，标准的两阶段 RAG 检索
设计，见 reranker.py 的说明）。"""

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

# RRF 融合排序之后，取排名前多少个候选交给重排序模型做精排——重排序模型
# 没法像向量检索那样提前把全部语料算好、查询来了只需要比对，每个候选都要
# 现跑一次模型推理（实测 bge-reranker-v2-m3 在这台机器上大概 100~120 毫秒
# 一对），对全部 9400 个 chunk 逐一打分要 15~38 分钟，扛不住，必须先用便宜
# 的向量+关键词+标签检索粗筛出一小批候选，只对这一小批做精细判断。
#
# 这一步的原则是"宁可多捞、不要漏掉"（检索阶段负责召回率，精确度交给重排序
# 阶段负责，这是检索增强生成领域的标准分工）——RRF 融合排序本身已经不再
# 对稠密检索的原始相似度做任何硬性门槛过滤（之前版本在这里卡了一个
# _MIN_DENSE_SCORE=0.5，撞到过真实反例："银行流水"这个查询里，真正相关的
# 发票原始相似度只有 0.4746，会被 0.5 的门槛误伤掉，而一份完全不相关的
# SQL migration 教程原始相似度有 0.545，反而能通过门槛——说明向量相似度
# 这个信号本身就不够可靠，硬卡阈值卡的位置很容易卡错，不如把"够不够相关"
# 这个判断完全交给下面更准的重排序模型，检索阶段只管把候选面撒宽一点）。
#
# 80 这个数字是参考"两阶段检索：向量捞前 50~100，精排到前几个"这个业界
# 惯用做法定的，不是精确算出来的——多捞一些，重排序的耗时会跟着线性涨
# （每对约 100~120 毫秒，80 个大概 8~10 秒），是候选池大小和查询延迟之间
# 的权衡，可以用 --candidates 参数调。
DEFAULT_CANDIDATE_POOL_SIZE = 80

# 重排序模型给出的最低相关度门槛——原始分数（不是 0~1 的概率，见
# reranker.py 里 LocalReranker.score() 为什么不用 sigmoid 的说明）低于这个
# 值的候选，不算"真正相关"，直接过滤掉。
#
# 真正相关的案例分数在 -2.6~+0.5 之间（"投资建议"-1.384、"工程师思维"
# -2.637、"不动产登记"+0.534——分数越高越相关，+0.534 是最贴切的一次，
# 不动产查询单本身就是查询词的完美对应）。
#
# 第一版按"完全不相关"的极端案例（"银行流水"对应的 SQL migration 教程
# -10.945、"露营装备推荐"对应的一份不相关发票 -10.984）定过 -5.0，实测
# 直接暴露了真实回归：接入候选池上限之后重新测"露营装备推荐"（语料里
# 确实没有这个话题），冒出了 19 个分数在 -3.4~-3.55 之间、明显不相关的
# 结果（一个 App 版本号 JSON、一份 Nest 教程、一段 base64 编码）——说明
# "完全不相关"不是只会落在 -11 这种极端值附近，也会落在 -3~-4 这个更
# "暧昧"的中间地带，只用两个最极端的反例校准，覆盖不到这个区间。改成
# -2.5（卡在已知最差的真正相关案例 -2.637 附近，留一点点余量）之后重测：
# "露营装备推荐"正确清零，之前验证过的"报销发票"（62 个通过，top 3 全对）/
# "工程师思维"（9 个通过，排序对）/"不动产登记"（4 个通过，top1 精确命中）/
# "微服务"（19 个通过，top 3 全对）都没有被误伤。
#
# 这依然只是拿有限样本校准出来的估计值，跟 _MIN_DENSE_SCORE 当初"只有
# 4 个查询词的证据"是同一类性质，换 reranker 模型、或者语料话题分布发生
# 大幅变化，需要重新用真实查询词测一遍——这次踩的坑已经说明"边界案例
# 只测最极端的两头是不够的"，以后调这个数字要多测几个"确实不相关但也
# 不是特别离谱"的中间地带案例，不能只看最好和最差两个极端。
_MIN_RERANK_SCORE = -2.5


def search_documents(
    query,
    embedder,
    reranker,
    db_path=DEFAULT_DB_PATH,
    top_k=5,
    candidate_pool_size=DEFAULT_CANDIDATE_POOL_SIZE,
):
    """两阶段检索：先粗筛，后精排。

    阶段一（粗筛，负责召回率，宁可多捞、不要漏掉）：向量语义相似度 +
    关键词精确匹配（FTS5 + bm25）+ 标签关键词匹配（同样是 FTS5 + bm25，
    但匹配的是文档整体的标签，不是某个 chunk）三路结果按排名融合
    （Reciprocal Rank Fusion），不是按原始分数加权——三路分数的数值尺度
    完全不是一回事（余弦相似度是 0~1 的有界值，bm25 是跟语料规模有关的
    无界值），硬要按百分比加权需要手调系数，而且没有验证数据支撑，不
    可靠。RRF 只看各自的排名，不比较原始分数，天然公平。融合排序取前
    `candidate_pool_size` 个，作为送去精排的候选池，这一步的排名/分数
    只用来"选哪些候选"，不是最终展示用的分数。

    标签这一路存在的意义：向量/关键词检索都是按 chunk 粒度算分的，一份文档
    整体的主题如果没有集中体现在某一个 800 字符的 chunk 里，会在那两路检索
    里"隐形"——但这个主题恰恰是打标签时模型看完整篇之后才提炼出来的，标签
    检索能补上这个盲区（真实案例：一份带着"工程师思维"这个标签的文档，
    在向量+关键词两路混合检索里排到了第 2 名，第 1 名是一份没有这个标签、
    只是话题沾边的文档）。

    关键词/标签这两路都是字符三元组匹配（trigram），不是真正的分词，查询词
    至少要 3 个字符才能命中——2 字短查询这两路都会搜不到东西，这是已知的
    设计代价，不是 bug，向量检索没有长度限制，能接住这类短查询。

    阶段二（精排，负责精确度）：候选池交给 cross-encoder 重排序模型
    （reranker.py 的 LocalReranker），查询和每个候选的实际内容做真正的
    交互式判断，用重排序自己的分数（见 _MIN_RERANK_SCORE）决定最终排序、
    以及"够不够真的相关"——不再依赖粗筛阶段任何一路的原始分数去判断相关性。

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

    candidates = _rrf_merge(dense_ranking, sparse_ranking, tag_ranking)[:candidate_pool_size]

    t4 = time.monotonic()
    reranked_all = _rerank(reranker, query, candidates)  # 完整排序，还没过滤
    rerank_seconds = time.monotonic() - t4

    reranked = [r for r in reranked_all if r.score >= _MIN_RERANK_SCORE]
    results = reranked[:top_k]

    # 诊断信号，不参与任何判断逻辑——第一名和第二名的分差（gap）。
    #
    # 本来是想验证"检索增强生成领域常提的 gap 信号（比如 TARG 那类工作）能不能
    # 替代/补充 _MIN_RERANK_SCORE 这种固定阈值"，用 tests/run_search_regression.py
    # 跑过 13 个真实案例后，**已经拿到明确的反证，这条路径不可行**：
    # `宠物疫苗接种记录`（该判定不相关的假阳性）gap 反而很大（1.674，因为
    # 它虽然分数低，但比全库倒数第二名还是拉开了距离）；`日本签证`/`财务凭证`
    # （真正该判定相关、只是好答案不止一个）gap 反而接近 0。gap 离不开"绝对
    # 分数本身够不够高"这个前提，脱离绝对分数单独用会把结论判反，不是"数据还
    # 不够多"，是这个信号本身在我们的真实数据上不具备独立判别力。
    #
    # 继续保留这两个字段是因为它们现在确实在被用——tests/run_search_regression.py
    # 每次跑回归集都会打印，是留存下来的真实观测数据，不是猜想着"以后可能用得上"
    # 的死代码；只是不要再假设未来会拿它当正式判断依据，除非出现新的、有说服力
    # 的证据。
    top1_top2_gap = None
    if len(reranked_all) >= 2:
        top1_top2_gap = reranked_all[0].score - reranked_all[1].score
    elif len(reranked_all) == 1:
        top1_top2_gap = float("inf")  # 只有一个候选，没有"第二名"可比，视为无穷大的分差

    stats = {
        "query_embed_seconds": query_embed_seconds,
        "rerank_seconds": rerank_seconds,
        "candidate_pool_size": len(candidates),
        "vector_search_seconds": dense_seconds,
        "keyword_search_seconds": sparse_seconds,
        "tag_search_seconds": tag_seconds,
        "chunk_count": chunk_count,
        "top1_score": reranked_all[0].score if reranked_all else None,
        "top1_top2_gap": top1_top2_gap,
        "document_count": len(reranked),
        "dense_hit_count": len(dense_ranking),
        "sparse_hit_count": len(sparse_ranking),
        "tag_hit_count": len(tag_ranking),
    }

    return results, stats


def _dense_search(index, model_id, query_vector, normalized):
    """向量检索：把全部 chunk 向量一次性拼成一个矩阵，跟 query 向量做一次批量
    矩阵乘法算余弦相似度，不是之前那种逐行 Python 循环手算点积再比较——语料
    一大，矩阵运算比 Python 循环快得多。

    同一个文档命中好几个 chunk 时只保留分数最高的那个，返回按分数从高到低
    排好序的全部文档——不在这里对原始相似度做任何硬性门槛过滤（早期版本
    在这里卡过 _MIN_DENSE_SCORE=0.5，已经撤掉，见上面 DEFAULT_CANDIDATE_
    POOL_SIZE 的注释）。这一路只负责"粗筛、尽量别漏掉"，"够不够真的相关"
    这个判断完全交给后面的重排序模型。
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


def _rerank(reranker, query, candidates):
    """对 RRF 粗筛出来的候选池做精排：每个候选原来展示用的那段文本
    （chunk 正文，或者标签命中给的标签原文）拿去跟查询一起喂给重排序
    模型，重排序自己给出的分数直接替换掉候选原本的 RRF 分数（不再是
    "排名贡献值的累加"这种没有直接意义的数字，是重排序模型真正判断出来
    的"这俩有多相关"），按这个新分数从高到低排序。

    返回**完整**的排好序的列表，不在这里过滤——过滤（_MIN_RERANK_SCORE）
    挪到调用方做，是因为回归测试脚本需要看到完整排序（包括没通过门槛的
    候选），才能算出"第一名和第二名分差多大"这类诊断信号，如果在这里就
    把没通过门槛的候选丢掉，这个信号就没法算了。
    """
    if not candidates:
        return []

    texts = [c.text for c in candidates]
    scores = reranker.score(query, texts)

    scored = [
        SearchResult(path=c.path, score=score, chunk_index=c.chunk_index, text=c.text)
        for c, score in zip(candidates, scores)
    ]
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored


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
