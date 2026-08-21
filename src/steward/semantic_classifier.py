"""端侧文档打标签引擎 V2 —— 逐文件直接打标签。

标签（tags）是验证过可靠、真正对外的能力；分类（category）保留，但降级为内部参考
信号（用来减少 SLM 调用量、辅助归档），不追求对用户展示的整洁度。每份文档独立调用
一次本地 SLM，产出 category + tags + reasoning，不再依赖聚类簇的集体判断（V1 的
Cluster-then-Label 设计已废弃，为什么废弃、V2 具体怎么设计，见
docs/dynamic_classification_architecture.md，这里只放实现）。

图片/视频/代码/压缩包等非文档类型、以及提取失败的文件，不进入下面这条 SLM 语义
判断的管线，只在最后补一道基础类型标签（见 Stage②-basic），不是被忽略。

四阶段管线：
  ① 内容量预筛 —— 文本过短/无信息量，直接判 unclassified，不调用 SLM
  ② 分两条路：
     - 大文件夹先判断整体连贯性——只看文件夹名字+文件名列表（不读内容），一次调用
       判断这个文件夹是不是一门课程/一个项目这种"一件事"，命名规律（统一编号+标题
       格式）是依据，不要求内部每个文件话题完全一致。判定连贯：整个文件夹直接沿用
       这一次判断的结果，不再逐个调用 SLM。（最早试过"抽样几个文件独立判断，看
       判断结果收不收敛"，实测会被文件夹内部正常的话题差异误判成不连贯——同一门
       课程里前端/后端/DevOps 课时话题本来就不一样——所以改成直接问文件夹本身）
     - 判定不连贯 / 文件数不够多的文件夹：逐文件独立调用 SLM，产出各自的
       category + tags + reasoning，互不知道彼此的判断。
     用户真正要的是"整理一个指定目录"，不完全等于"给每个文件独立发现语义"——
     目录里已经有的结构本身是一份强先验，该直接利用，不是每次都当作不存在。
     之前试过给逐文件判断加"已用分类名，优先复用"的收敛压力，两份不同语料上都
     测出来会让内容被硬塞进不相关的已有分类，代价不可接受，已经撤掉——这次的
     文件夹整体判断不一样，没有对任何一次判断施加压力，只是换了一个更合适的
     判断对象（文件夹本身，不是拿逐文件判断结果反推）
  ③ Taxonomy 归一化 —— 合并同义分类名（对象是分类名字符串，不涉及向量），
     只是缓解分类名数量偏多这个"不够整洁"的问题，不承担保证判断准确的责任
  ④ 置信度计算 —— 文档向量跟分类名/标签向量做点积，作为参考信息存下来；
     实测过"跟候选池里所有分类名比排名，不是第一就打回 unclassified"这道复核，
     候选分类名一多（30+）就会被近义词的噪声大量误伤，已经撤掉，直接信 Stage②
     里 SLM 自己的判断（这个判断本身已经用真实案例验证过是可靠的）
  ⑤ 持久化到 SQLite
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
from openai import OpenAI


# Stage①：内容去空白后短于这个字符数，直接判 unclassified，不调用 SLM
MIN_CONTENT_LENGTH = 30

# Stage②：文件夹至少有这么多文件，才值得判断"整体是不是一个连贯的东西"（太小的
# 文件夹，直接逐文件判断也没几次调用，不用额外绕一道）。还没拿真实数据校准过，
# 是合理起步值，不是拍死的常量。
FOLDER_BATCH_MIN_SIZE = 8

# 并发调用 SLM 的线程数。这是推理引擎侧的配置，需要跟 llama-server 启动时的
# -np（slot 数）匹配，不要跟下面的分类逻辑耦合——后续要支持根据硬件自动探测时，
# 只需要改调用方传进来的这一个参数，不用碰管线逻辑本身。
DEFAULT_MAX_WORKERS = 8

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LLM_MODEL = "Qwen3.5-2B-Q4_K_M.gguf"


@dataclass
class ClassificationResult:
    """分类与打标签的标准输出契约。"""

    category: str
    tags: List[Tuple[str, float]]  # [(tag_name, confidence)]
    confidence: float
    status: str
    reasoning: str


def _build_llm_client(base_url: str = DEFAULT_LLM_BASE_URL) -> OpenAI:
    # trust_env=False 禁用代理对 localhost 的误拦截；timeout 防止服务无响应时无限阻塞
    return OpenAI(
        base_url=base_url,
        api_key="no-key-required",
        http_client=httpx.Client(trust_env=False, timeout=60.0),
    )


def _extract_snippet(text: str, file_path: str = "", max_len: int = 1500) -> str:
    """提取文档的有效正文片段，根据文件类型做差异化处理。

    - HTML 文件：先剥离标签，跳过前 200 字符的 <head>/导航区
    - PDF 文件：跳过前 350 字符（发票/表单类 PDF 前段通常是固定表头字段）
    - 其他（.md/.txt）：直接从头取，内容通常从第一行就有语义
    """
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""

    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()

    if ext in ("html", "htm"):
        start = 200 if len(clean) > 400 else 0
    elif ext == "pdf":
        start = 350 if len(clean) > 600 else 0
    else:
        start = 0

    return clean[start:start + max_len].strip()


def _classify_one_document(client: OpenAI, model_name: str, path: str, full_text: str) -> Optional[dict]:
    """调用一次 SLM，独立判断单份文档的分类。返回 {category, tags, reasoning}，解析失败返回 None。

    曾经试过给这个 prompt 加一段"已经用过的分类名，优先复用"的提示，用批次推进的方式让
    判断之间互相看得见——实测在两份完全不同的语料（整理过的 Documents、零散的 Downloads）
    上都验证过，这个做法会让模型为了凑到已有分类里，把明显不相关的内容硬塞进去（销售数据
    被归成"AI 模型评测报告"、英语学习进度被归成同一类），是拿准确性换分类数量好看，代价
    不可接受，已经撤掉。分类名数量偏多（几十个）是一个"不够整洁"的问题，靠 Stage③ 的
    归一化去缓解；判断本身的准确性没有退路，独立判断是目前唯一验证过内容不会串味的方式。
    """

    filename = os.path.basename(path)
    parent_dir = os.path.basename(os.path.dirname(path))
    snippet = _extract_snippet(full_text, path)

    # 静态说明文字全部放在前面、连成一段不变的前缀，变化的文件内容放最后——
    # prompt 缓存只认"从开头连续匹配到哪"，变化内容一旦出现在中间，后面哪怕是完全
    # 相同的文字也不能命中缓存了（KV cache 是按 token 顺序累积算出来的，后面的 token
    # 依赖前面全部 token 的上下文，前缀一旦分叉，后面就不再是"同一段计算"）。
    prompt = (
        "你是一个专业的端侧文件整理助手。请分析给定的文件，独立给出分类结论。\n\n"
        "要求：category(2-4字宏观主分类)、tags(3-6个细粒度关键词)、reasoning(一句话依据)。\n"
        "严格输出纯JSON，不要markdown标记：\n"
        '{"category": "...", "tags": ["...", "..."], "reasoning": "..."}\n\n'
        "待分析的文件如下：\n"
        f"所在目录: {parent_dir}\n文件名: {filename}\n内容片段: {snippet}"
    )

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是严格只返回JSON格式的文件分类助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = resp.choices[0].message.content.strip()
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            return None
        parsed = json.loads(content[start:end + 1])
        category = str(parsed.get("category", "")).strip()
        tags = [str(t).strip() for t in parsed.get("tags", []) if str(t).strip()]
        reasoning = str(parsed.get("reasoning", "")).strip()
        if not category:
            return None
        return {"category": category, "tags": tags, "reasoning": reasoning}
    except Exception as e:
        print(f"  [Warning] 分类失败 {filename}: {e}", flush=True)
        return None


# Stage③：分类名之间向量相似度超过这个值就认为是同义词，归为一组。
# 用真实分类名实测校准过：人工标注的"该合并"对点积落在 0.629~0.869，"不该合并"对
# 落在 0.406~0.524，中间有明显空隙，0.60 卡在这段空隙里。（之前直接拍了 0.80，
# 结果把"技术文档"和"技术笔记"这种一看就是同义词的也判成不同类，太严格。）
TAXONOMY_SIMILARITY_THRESHOLD = 0.60


def _normalize_taxonomy(
    categories: List[str],
    client: OpenAI,
    model_name: str,
    embedder,
    similarity_threshold: float = TAXONOMY_SIMILARITY_THRESHOLD,
    category_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, str]:
    """把一组分类名里语义重叠的合并为统一名称。

    分三步，不是一次性丢给 LLM 自己判断"这堆名字里哪些该合并"：
    1. 向量粗筛：算出每个分类名的向量，相似度超过阈值的贪心归为一组，作为"疑似
       同义词"的候选——"哪些名字可能是同一回事"先交给数学判断，不指望 LLM 自己去
       记住、比较几十上百个抽象字符串。实测过让 2B 小模型直接看着一批分类名自己
       判断分组，不仅会漏合并明显的同义词（前端开发/后端开发/全栈开发这类，几乎
       不合并），还会把语义完全不沾边的名字错误合并在一起（"证件合同"被并进过
       "技术文档"），不是分批就能解决的，是这类任务它本身就不擅长。
    2. LLM 核实+起名：向量相似度自己也不够干净——实测过"开发指南"和"出行指南"
       只是共享"指南"两个字，点积能到 0.7，跟真正的同义词数值区间是重叠的，
       没有一个阈值能把这两类情况完全分开。所以候选组还要过一道 LLM 复核：判断
       "这几个是不是真的一类，有没有字面像但语义不同混进来的"，这是简单的是非题，
       比"自己从零发现分组"可靠得多。**"起名字"这一步不再交给 LLM**——曾经试过
       把每个候选名代表多少个文件写进 prompt、明确要求"以文件数最多的为准"，实测
       2B 模型依然会被措辞更完整的少数派名字带偏（"Nest 技术课程"代表 200 个文件，
       混进来一份真正的 React Native 动画文件后，整组 200 个文件被改名成了只代表
       1 个文件的"React Native 动画技术"）。现在标准名直接在代码里取组内文件数
       最多的那个原始名字，不再指望模型"权衡"出正确结果。
    3. 落单的组（向量粗筛阶段就没找到相似的）直接沿用自己的名字，不用调用 LLM。

    :param category_counts: {分类名: 这个分类名当前代表多少个文件}，用来在代码里
        挑出每组的标准名。不传就当所有分类名代表 1 个文件（旧行为）。
    :return: {原始分类名: 归一化后的标准分类名}
    """
    if not categories:
        return {}
    if len(categories) == 1:
        return {categories[0]: categories[0]}

    counts = category_counts or {}
    groups = _group_by_similarity(categories, embedder, similarity_threshold)

    result: Dict[str, str] = {}
    for group in groups:
        if len(group) == 1:
            result[group[0]] = group[0]
        else:
            result.update(_verify_and_name_group(group, counts, client, model_name))

    return result


def _group_by_similarity(categories: List[str], embedder, threshold: float) -> List[List[str]]:
    """按向量相似度贪心分组：每个名字归到跟它最相似、且相似度过阈值的已有组；
    找不到就自己开一组。组的"代表向量"固定用这一组第一个成员的向量，不随后面加入
    的成员更新——保持简单，避免"组的中心跟着成员增多慢慢漂移，导致后面判断越来越
    松"这种不好排查的行为。
    """
    vectors = np.array(embedder.embed_documents(categories), dtype=np.float32)  # 已经归一化

    group_members: List[List[str]] = []
    group_rep_vectors: List[np.ndarray] = []

    for name, vec in zip(categories, vectors):
        best_group_idx = None
        best_sim = -1.0
        for idx, rep_vec in enumerate(group_rep_vectors):
            sim = float(np.dot(vec, rep_vec))
            if sim > best_sim:
                best_sim = sim
                best_group_idx = idx
        if best_group_idx is not None and best_sim >= threshold:
            group_members[best_group_idx].append(name)
        else:
            group_members.append([name])
            group_rep_vectors.append(vec)

    return group_members


def _verify_and_name_group(
    group: List[str], counts: Dict[str, int], client: OpenAI, model_name: str
) -> Dict[str, str]:
    """核实一组向量相似度筛出来的"疑似同义词"，剔除字面像但语义不同的。

    合并后的标准名**不再让 LLM 自由发挥去起新名字**，直接取组内代表文件数最多的那个
    原始名字——实测过：就算把每个候选名代表多少个文件明确写进 prompt、还专门叮嘱
    "以代表文件数量最多的为准"，2B 模型依然会被措辞更完整/更具体的少数派名字带偏
    （"Nest 技术课程"代表 200 个文件，"React Native 动画技术"只代表 1 个，结果整组
    200 个文件被改名成了后者）。"给一堆结构化数字、按权重做决策"这类任务，今天已经
    反复验证过 2B 模型不擅长；文件数这个数字本来就在我们自己手上，直接用代码决定，
    不再指望模型帮忙"权衡"出正确结果——LLM 只负责回答"这几个是不是真的一类"这个
    它比较擅长的是非题，不负责起名字。

    :param counts: {分类名: 代表多少个文件}，用来在代码里挑出该组的标准名。
    :return: {原始分类名: 归一化后的标准分类名}——被剔除的分类名映射到自己（不合并）
    """
    prompt = (
        f"以下几个分类名是根据向量相似度粗略挑出来的候选，可能包含误判（比如两个名字"
        f"只是字面相似，实际语义不同）：{'、'.join(group)}\n\n"
        "请判断哪些确实属于同一类（语义真正相同或高度重合，不是碰巧共享几个字）。\n"
        "严格输出纯 JSON，不要markdown标记：\n"
        '{"同一类的": ["...", "..."], "不属于的": ["..."]}\n'
        '如果全部都属于同一类，"不属于的"填空数组。'
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是严格只返回 JSON 的分类核实助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=len(group) * 20 + 100,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = resp.choices[0].message.content.strip()
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("未找到 JSON")
        parsed = json.loads(content[start:end + 1])
        confirmed = {str(c).strip() for c in parsed.get("同一类的", [])}
        confirmed &= set(group)  # 只信确实在候选组里的名字，防止 LLM 编造/拼错

        if not confirmed:
            return {c: c for c in group}

        # 标准名 = 被确认属于同一类的成员里，代表文件数最多的那个原始名字。
        canonical_name = max(confirmed, key=lambda c: counts.get(c, 1))

        result: Dict[str, str] = {}
        for c in group:
            # 只信 LLM 明确确认属于同一类的；没提到的、或明确说不属于的，都不合并，
            # 保留自己的名字——宁可少合并，不要把语义不同的名字硬凑成一类。
            result[c] = canonical_name if c in confirmed else c
        return result
    except Exception as e:
        print(f"  [Warning] 分类组核实失败 ({e})，这组不合并，各自保留原名。", flush=True)
        return {c: c for c in group}


# Stage②-a：判断一个大文件夹整体是不是一个连贯的整体时，最多把文件名列表喂给模型看
# 多少个——文件名的命名规律（统一编号+标题格式）是判断依据，不需要看完全部文件名，
# 列太多反而占用不必要的 token。
FOLDER_VERDICT_MAX_FILENAMES = 30


def _classify_folder(client: OpenAI, model_name: str, folder_name: str, filenames: List[str]) -> Optional[dict]:
    """只根据文件夹名字 + 文件名列表，判断这个文件夹整体是不是一个连贯的整体（一门课程、
    一个项目、一批同类型资料），不读取任何文件内容，一次调用覆盖整个文件夹，跟文件夹
    里有多少个文件无关。

    这跟"抽样几个文件，独立判断，看判断结果收不收敛"是两个不同的问题——后者拿"文件夹
    内各文件具体话题一不一致"当信号，同一门课程里前端/后端/DevOps 话题本来就会不一样，
    会被误判成不连贯（实测过，`Nest 通关秘籍` 这种典型的"一门课"就是这样被误判的）。
    这次直接问"这个文件夹本身是不是一件事"，命名规律才是该看的信号，不是内部话题够不够
    一致。返回 None 代表判定不连贯（或者判断失败），调用方应该老实退回逐文件独立判断。
    """
    shown = filenames[:FOLDER_VERDICT_MAX_FILENAMES]
    name_list_str = "\n".join(f"- {n}" for n in shown)
    more_note = (
        f"\n（文件夹内共 {len(filenames)} 个文件，以上只列出前 {FOLDER_VERDICT_MAX_FILENAMES} 个）"
        if len(filenames) > FOLDER_VERDICT_MAX_FILENAMES else ""
    )

    prompt = (
        "你是一个专业的端侧文件整理助手。请只根据文件夹名字和里面的文件名列表判断，"
        "不需要读取任何文件的实际内容。\n\n"
        "判断标准：重点看文件名之间有没有统一的结构/命名规律（比如统一的编号+标题格式、"
        "统一的日期前缀），这种结构性规律说明这些文件是同一门课程/同一个项目按顺序产出的，"
        "才是真正的\"连贯\"——这种情况下不要求每个文件具体话题完全一致（比如一门技术课程，"
        "有的课时讲前端、有的讲后端、有的讲部署，属于正常的同一件事的不同部分）。\n\n"
        "不要仅仅因为几个文件的话题可以被笼统地归到同一个抽象大类下（比如\"都跟个人成长"
        "有关\"、\"都是学习记录\"）就判定为连贯——如果文件名之间没有统一的结构/命名规律，"
        "每个文件名描述的是各自独立的、具体的事件或话题（比如不同次聊天各自聊的完全不同"
        "内容、互不相关的零散笔记），应该判定为不连贯，交给逐文件单独判断，不要牵强地找一个"
        "笼统的主题把它们框在一起。\n\n"
        "举例：\n"
        "「1. xxx.md」「2. yyy.md」「3. zzz.md」这种统一编号格式 → 连贯，是一个系列/课程\n"
        "「与Gemini聊xxx的话题.txt」「关于yyy的investment对话.txt」这种各自独立命名、"
        "内容各不相同的零散记录 → 不连贯，即使都能笼统归为\"个人笔记/聊天记录\"\n\n"
        f"文件夹名: {folder_name}\n"
        f"文件名列表:\n{name_list_str}{more_note}\n\n"
        "严格输出纯JSON，不要markdown标记：\n"
        '{"is_coherent": true/false, "category": "2-4字分类名(is_coherent为true时必填)", '
        '"tags": ["3-6个细粒度关键词(is_coherent为true时必填)"], "reasoning": "一句话依据"}'
    )

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是严格只返回JSON格式的文件夹整理助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = resp.choices[0].message.content.strip()
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            return None
        parsed = json.loads(content[start:end + 1])
        if not parsed.get("is_coherent"):
            return None
        category = str(parsed.get("category", "")).strip()
        if not category:
            return None
        tags = [str(t).strip() for t in parsed.get("tags", []) if str(t).strip()]
        reasoning = str(parsed.get("reasoning", "")).strip()
        return {"category": category, "tags": tags, "reasoning": reasoning}
    except Exception as e:
        print(f"  [Warning] 文件夹整体判断失败 ({folder_name}): {e}", flush=True)
        return None


def run_dynamic_classification_pipeline(
    index,
    embedder=None,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    min_content_length: int = MIN_CONTENT_LENGTH,
) -> dict:
    """V2 端到端逐文件分类管线，见模块顶部的四阶段说明。"""

    from steward.document_vectors import get_all_document_vectors
    from steward.embeddings import LocalEmbedder

    if embedder is None:
        embedder = LocalEmbedder()

    client = _build_llm_client(llm_base_url)

    print("[Step 0] 清空旧有分类数据...", flush=True)
    index.clear_classifications()

    print("[Step 1] 加载已提取文本的文档...", flush=True)
    rows = index.connection.execute(
        """
        SELECT d.id AS document_id, d.path AS path, x.full_text AS full_text
        FROM documents d JOIN extractions x ON x.document_id = d.id
        WHERE d.is_present = 1 AND x.status = 'success'
        """
    ).fetchall()
    print(f"[Step 1] 共 {len(rows)} 份文档。", flush=True)

    # Stage①：内容量预筛
    to_classify = []
    skipped_short = []
    for row in rows:
        text = (row["full_text"] or "").strip()
        if len(text) < min_content_length:
            skipped_short.append(row["document_id"])
        else:
            to_classify.append(row)
    print(f"[Step 1] 内容过短跳过 {len(skipped_short)} 份，{len(to_classify)} 份进入 SLM 分类。", flush=True)

    # Stage②-a：大文件夹整体连贯性判断。只看文件夹名字+文件名列表，不读任何文件内容，
    # 一个文件夹一次调用就能判断完，跟文件夹里有多少个文件无关——文件夹本身是用户
    # 已经整理好、起了名字的一个整体，不该被拆开当成一堆互不相干的文件重新发现一遍。
    # 判定连贯：整个文件夹直接沿用这一次判断的 category + tag 候选池，不需要再对
    # 里面的文件逐个独立调用 SLM。判定不连贯：老实退回逐文件独立判断，不强行合并。
    folder_groups: Dict[str, list] = {}
    for row in to_classify:
        folder_groups.setdefault(os.path.dirname(row["path"]), []).append(row)

    large_folders = {f: rs for f, rs in folder_groups.items() if len(rs) >= FOLDER_BATCH_MIN_SIZE}
    folder_verdict: Dict[str, dict] = {}  # {folder: {"category":..., "tags":[...], "reasoning":...}}

    if large_folders:
        print(f"[Step 2] {len(large_folders)} 个文件夹文件数≥{FOLDER_BATCH_MIN_SIZE}，判断整体连贯性...", flush=True)
        for folder, rs in large_folders.items():
            filenames = [os.path.basename(row["path"]) for row in rs]
            verdict = _classify_folder(client, llm_model, os.path.basename(folder), filenames)
            if verdict:
                folder_verdict[folder] = verdict
                print(f"  [Step 2] 「{os.path.basename(folder)}」({len(rs)} 个文件) 判定连贯，归为「{verdict['category']}」", flush=True)

    # Stage②-b：不在已判定连贯文件夹里的文件，走逐文件独立判断（跟之前完全一样）。
    to_call = [row for row in to_classify if os.path.dirname(row["path"]) not in folder_verdict]
    print(
        f"[Step 2] {max_workers} 路并发调用 SLM（{len(to_call)} 份逐文件判断，"
        f"{len(to_classify) - len(to_call)} 份沿用文件夹整体判断）...", flush=True
    )
    raw_results: Dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_classify_one_document, client, llm_model, row["path"], row["full_text"]): row["document_id"]
            for row in to_call
        }
        done = 0
        for future in as_completed(futures):
            doc_id = futures[future]
            result = future.result()
            if result:
                raw_results[doc_id] = result
            done += 1
            if done % 50 == 0 or done == len(to_call):
                print(f"  [Step 2] 已完成 {done}/{len(to_call)}", flush=True)

    # Stage②-c：已判定连贯的文件夹，文件直接沿用文件夹整体判断的 category + tag 候选池。
    # tags 是共享候选池，不是每个文件死板地拿到一模一样的标签——具体哪些标签打得上、
    # 打多高的分，还是靠 Stage④ 里各自文档向量跟候选标签向量做点积决定，同一个候选池，
    # 讲 React 的课时和讲数据库的课时天然会打出不一样的标签分数，不需要额外调用 SLM。
    for row in to_classify:
        if row["document_id"] in raw_results:
            continue
        verdict = folder_verdict.get(os.path.dirname(row["path"]))
        if verdict:
            raw_results[row["document_id"]] = {
                "category": verdict["category"],
                "tags": verdict["tags"],
                "reasoning": f"文件夹「{os.path.basename(os.path.dirname(row['path']))}」整体判定：{verdict['reasoning']}",
            }

    parse_failed = [row["document_id"] for row in to_classify if row["document_id"] not in raw_results]
    print(f"[Step 2] 分类完成：{len(raw_results)} 份成功，{len(parse_failed)} 份解析失败。", flush=True)

    # Stage③：Taxonomy 归一化（对象是分类名字符串，不涉及向量）
    print("[Step 3] Taxonomy 归一化...", flush=True)
    category_counts: Dict[str, int] = {}
    for r in raw_results.values():
        category_counts[r["category"]] = category_counts.get(r["category"], 0) + 1
    raw_categories = sorted(category_counts.keys())
    canonical_map = _normalize_taxonomy(
        raw_categories, client, llm_model, embedder, category_counts=category_counts
    )
    for r in raw_results.values():
        r["category"] = canonical_map.get(r["category"], r["category"])
    canonical_categories = sorted(set(canonical_map.values()))
    print(f"[Step 3] {len(raw_categories)} 个原始分类名归一化为 {len(canonical_categories)} 个。", flush=True)

    # Stage④：置信度计算（category 和 tag 用同一套数学：文档向量分别跟候选文本的
    # 向量做点积）。这个分数只是存下来给人参考，不再用来推翻 Stage② 里 SLM 自己
    # 判定的分类——为什么不拿它去做"复核"，见上面模块顶部说明。
    print("[Step 4] 计算文档向量与分类名/标签向量的点积置信度...", flush=True)
    doc_vectors_map = get_all_document_vectors(index)
    final_results: Dict[int, ClassificationResult] = {}

    def _score_tags(raw_tags: List[str], doc_vec) -> List[Tuple[str, float]]:
        # 每份文档自己的标签现场单独 embed，矩阵行顺序天然跟 raw_tags 顺序一致，
        # 不需要维护任何"标签名 → 行号"的查表结构。embed_documents() 出来的向量
        # 已经是单位向量（见 embeddings.py 的 normalize_embeddings=True），不用再手动归一化。
        if doc_vec is None or not raw_tags:
            return [(t, 0.5) for t in raw_tags]
        tag_matrix = np.array(embedder.embed_documents(raw_tags), dtype=np.float32)
        sims = np.clip(tag_matrix @ doc_vec, 0.0, 1.0)
        # [("前端", 0.61), ("后端", 0.58), ("配置", 0.52)]——一个装着 (标签名, 分数) 元组的普通列表
        return list(zip(raw_tags, sims.tolist()))

    if canonical_categories and raw_results:
        # 同样，embed_documents() 出来的向量已经归一化过，这里不用再手动做一遍。
        category_matrix = np.array(embedder.embed_documents(canonical_categories), dtype=np.float32)
        category_index = {name: i for i, name in enumerate(canonical_categories)}

        for doc_id, r in raw_results.items():
            doc_vec = doc_vectors_map.get(doc_id)
            assigned_category = r["category"]
            scored_tags = _score_tags(r["tags"], doc_vec)

            if doc_vec is None or assigned_category not in category_index:
                final_results[doc_id] = ClassificationResult(
                    category="unclassified", tags=scored_tags, confidence=0.0,
                    status="unclassified", reasoning="缺少文档向量或分类名未成功归一化",
                )
                continue

            # 曾经在这里加过"跟候选池里所有分类名比排名，不是第一就打回 unclassified"
            # 的复核逻辑，实测发现是这次分类质量下降的真正原因：候选分类名有 30~50+
            # 个的时候，总会有某个语义相近的分类名偶然点积更高一点点，绝大多数被打回
            # unclassified 的文档，差距都只有 0.03~0.15 这种噪声级别，不是真判错了。
            # SLM 直接判断这件事本身已经用真实案例验证过是可靠的（包括简历、聊天记录
            # 这些以前 V1 会判错的案例）；不可靠的是这道"复核"，所以撤掉，改为直接信
            # SLM 的判断。点积依然算出来存着，当一个信息量参考，不再用来推翻分类结果。
            sims = np.clip(np.dot(category_matrix, doc_vec), 0.0, 1.0)
            assigned_idx = category_index[assigned_category]
            assigned_sim = float(sims[assigned_idx])

            final_results[doc_id] = ClassificationResult(
                category=assigned_category, tags=scored_tags, confidence=assigned_sim,
                status="classified", reasoning=r["reasoning"],
            )

    # 内容过短 / SLM 解析失败的，统一判 unclassified
    for doc_id in skipped_short:
        final_results[doc_id] = ClassificationResult(
            category="unclassified", tags=[], confidence=0.0,
            status="unclassified", reasoning="文本内容过短，跳过 SLM 分类",
        )
    for doc_id in parse_failed:
        final_results[doc_id] = ClassificationResult(
            category="unclassified", tags=[], confidence=0.0,
            status="unclassified", reasoning="SLM 输出 JSON 解析失败",
        )

    # Stage②-basic：非文档类型 / 没有成功提取过文本的文件（图片、视频、代码、压缩包……），
    # 目前完全不在这条流水线的语义分析范围内。以前这批文件"安静地不出现"，容易被误以为
    # 是遗漏；这里显式给一个浅层结果——不调用 SLM，直接用 Week 1 已经算好的 basic_type
    # 生成一个基础标签，status 单独标成 "basic"，跟真正走过 SLM 语义判断的结果区分开，
    # 讲清楚"这批文件只做了类型识别，没有做内容理解"，比彻底不处理要诚实。
    basic_rows = index.connection.execute(
        """
        SELECT d.id AS document_id, d.path AS path, d.basic_type AS basic_type
        FROM documents d
        WHERE d.is_present = 1
          AND d.id NOT IN (SELECT document_id FROM extractions WHERE status = 'success')
        """
    ).fetchall()
    for row in basic_rows:
        basic_type = row["basic_type"] or "unknown"
        ext = os.path.splitext(row["path"])[1].lstrip(".").lower() or "无扩展名"
        final_results[row["document_id"]] = ClassificationResult(
            category=f"{basic_type}文件",
            tags=[(basic_type, 1.0), (ext, 1.0)],
            confidence=0.0,
            status="basic",
            reasoning="非文档类型或未成功提取文本，仅做基础类型识别，未做语义内容分析",
        )
    if basic_rows:
        print(f"[Step 2-basic] {len(basic_rows)} 份非文档类型/提取失败的文件，完成基础标签。", flush=True)

    # Stage⑤：持久化
    print("[Step 5] 持久化写入 SQLite...", flush=True)
    with index.connection:
        for doc_id, result in final_results.items():
            index.save_classification(
                document_id=doc_id,
                category_name=result.category,
                tags=result.tags,
                confidence=result.confidence,
                status=result.status,
                reasoning=result.reasoning,
            )

    classified_count = sum(1 for r in final_results.values() if r.status == "classified")
    basic_count = sum(1 for r in final_results.values() if r.status == "basic")
    unclassified_count = len(final_results) - classified_count - basic_count

    print(
        f"🎉 完成！共 {len(final_results)} 份，{classified_count} 份深度打标签，"
        f"{basic_count} 份基础类型识别，{unclassified_count} 份未归类。",
        flush=True,
    )

    return {
        "total_documents": len(rows),
        "classified_count": classified_count,
        "basic_count": basic_count,
        "unclassified_count": unclassified_count,
        "skipped_short_count": len(skipped_short),
        "parse_failed_count": len(parse_failed),
        "canonical_categories": len(canonical_categories),
    }
