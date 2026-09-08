"""本地重排序（reranker）模型封装。

跟 embeddings.py 里的 LocalEmbedder 是同一类"本地模型封装"，但解决的是
不同问题，两者不是二选一，是标准的"先粗筛后精排"两阶段设计里的两个阶段：

- LocalEmbedder 是双塔结构（bi-encoder）——查询和文档分别独立编码成向量，
  文档这边的向量在建索引的时候就能提前算好、存进数据库，查询来了之后只
  需要现算一次查询向量，再跟数据库里已经算好的向量做一次矩阵乘法比对。
  快，代价是查询和文档之间没有任何"交互"，只能看两个向量在向量空间里
  离得近不近，看不出词与词之间真正的语义关系（真实撞过的坑：查询"银行
  流水"，向量检索把一份讲数据库 migration 教程的文档排到了比真正相关的
  发票还高，因为教程里一堆时间戳/数字表面上"看起来像"流水记录）。

- LocalReranker（这个模块）是 cross-encoder——查询和候选文档必须放在一起
  同时喂给模型做 attention 交互，每个词能"看见"对方的每个词，没法像双塔
  那样提前算好、离线存着，每次查询都要对每一个候选现算一次，慢得多（实测
  bge-reranker-v2-m3 在这台机器上大概 100~120 毫秒一对），但判断"这俩到底
  有多相关"这件事准得多——同一个"银行流水"案例，重排序给那份不相关的教程
  打了 -10.9 分，给真正相关的内容打了 -2~0 分左右，缺口很大很干净。

正因为 cross-encoder 没法提前算好、必须对每个候选现跑一次模型，成本是跟
"要比对的候选数量"成正比的，扛不住对全部语料（这个项目现在是 9400 个
chunk）逐一打分——测过：全部打一遍在这台机器上要 15~38 分钟。所以标准
做法是分两阶段：LocalEmbedder 配合关键词/标签检索，先从全部语料里快速
（几十毫秒到一秒级别）捞出一小批候选（现在是几十个量级），这个模块只对
这一小批候选做精细判断，成本才付得起。
"""

from pathlib import Path


# reranker.py 位于 src/steward/ 下，向上两层就是项目根目录。
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "bge-reranker-v2-m3"


class LocalReranker:
    """使用 FlagEmbedding 在本机对 (query, 候选文本) 打相关性分数。"""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, use_fp16=True):
        # 延迟导入：只有真正创建 LocalReranker 时才加载这个几百 MB 的模型库
        # 和权重，跟 LocalEmbedder 的做法一致。
        from FlagEmbedding import FlagReranker

        self._reranker = FlagReranker(str(model_path), use_fp16=use_fp16)

    def score(self, query, texts):
        """给一个查询和一批候选文本打分，返回原始分数（logit）列表，顺序
        跟传入的 texts 一一对应。

        故意不用 normalize=True 转成 0~1 的 sigmoid 概率——真实校准过
        （见 semantic_search.py 里 _MIN_RERANK_SCORE 的注释）：真正相关的
        案例原始分数在 -2.6~+0.5 之间，完全不相关的案例稳定卡在 -11 附近，
        中间有一个很大、很清晰的缺口，原始分数已经足够分辨，sigmoid 压缩
        到 0~1 反而会把这批案例全部挤到 0 附近（sigmoid(-11) 已经约等于
        0.00002），看不出真正相关和完全不相关之间那个有意义的差距。
        """
        if not texts:
            return []
        pairs = [[query, text] for text in texts]
        scores = self._reranker.compute_score(pairs)
        # 只有一对候选时，FlagEmbedding 部分版本会返回单个标量而不是列表，
        # 这里统一成列表，调用方不用关心这种边界情况。
        if not isinstance(scores, list):
            scores = [scores]
        return scores
