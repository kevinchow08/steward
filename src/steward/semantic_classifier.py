"""端侧文档打标签引擎 V2 —— 逐文件直接打标签。

标签（tags）是验证过可靠、真正对外的能力；分类（category）保留，但降级为内部参考
信号（用来减少 SLM 调用量、辅助归档），不追求对用户展示的整洁度。每份文档独立调用
一次本地 SLM，产出 category + tags + reasoning，不再依赖聚类簇的集体判断（V1 的
Cluster-then-Label 设计已废弃，为什么废弃、V2 具体怎么设计，见
docs/dynamic_classification_architecture.md，这里只放实现）。

图片/视频/代码/压缩包等非文档类型、以及提取失败的文件，不进入下面这条 SLM 语义
判断的管线，只在最后补一道基础类型标签（见 Stage⑥），不是被忽略。

管线（曾经有一版"先看文件夹名字+文件名列表判断整体连贯性，猜对了就跳过内容分析"
的分支，已经彻底撤掉——反复用真实数据验证下来，不读内容纯猜连贯性这件事本身
不可靠：文件夹名字信息量不够会猜错，多层嵌套的文件夹（一门课分好几层子章节）
因为每层单独看文件数都不够门槛，会被拆散成互不相关的好几类，属于系统性风险，
不是靠调参数能修的。现在每一份文件都真实读取内容独立判断，代价是 SLM 调用量
变大，换来的是不再有"完全没读内容就下结论"这种风险）：
  ① 内容量预筛 —— 文本过短/无信息量，直接判 unclassified，不调用 SLM
  ② 逐文件独立判断 —— 每份文件都基于真实内容独立调用一次 SLM，产出各自的
     category + tags + reasoning，互不知道彼此的判断。之前试过给这一步加
     "已用分类名，优先复用"的收敛压力，两份不同语料上都测出来会让内容被硬塞
     进不相关的已有分类，代价不可接受，已经撤掉，保持完全独立判断。
  ③ 同文件夹内标签统一 —— 每份文件都是基于真实内容独立判断出来的，但同一个
     文件夹里的文件，判断结果可能只是措辞不同、说的是同一件事。按文件夹分组，
     复用 Stage④ taxonomy 归一化的同一套安全网机制（向量粗筛+LLM核实+代码按
     数量选标准名），只是把范围从"全库"缩小到"同一个文件夹内"——像才合并，
     不像的（文件夹本来就是散装的）不强行合并。这一步不区分文件夹嵌套了几层，
     判断范围永远是"文件"和它的"直接父目录"这两级，不需要判断该在哪一层看，
     嵌套多深都一样处理。
  ④ Taxonomy 归一化 —— 在全库范围再合并一次同义分类名（对象是分类名字符串，
     不涉及向量），只是缓解分类名数量偏多这个"不够整洁"的问题，不承担保证
     判断准确的责任
  ⑤ 置信度计算 —— 文档向量跟分类名/标签向量做点积，作为参考信息存下来；
     实测过"跟候选池里所有分类名比排名，不是第一就打回 unclassified"这道复核，
     候选分类名一多（30+）就会被近义词的噪声大量误伤，已经撤掉，直接信 Stage②
     里 SLM 自己的判断（这个判断本身已经用真实案例验证过是可靠的）
  ⑥ 非文档类型基础标签 —— 图片/视频/代码/压缩包等不进入上面语义判断的文件，
     用 Week 1 已经算好的 basic_type 补一道浅层标签
  ⑦ 持久化到 SQLite
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
    不可接受，已经撤掉。分类名数量偏多（几十个）是一个"不够整洁"的问题，靠 Stage④ 的
    归一化去缓解；判断本身的准确性没有退路，独立判断是目前唯一验证过内容不会串味的方式。

    JSON 字段顺序特意让 reasoning 排在 category/tags 前面：真实语料上发现过几例
    category 跟 reasoning 完全对不上的案例（reasoning 明明写着"这是一份简历"，
    category 却填了"教育课程"）——LLM 是从左到右逐 token 生成的，先写的内容没法
    根据后面才写的内容调整，如果 schema 让模型先承诺 category、再补 reasoning，
    下结论的时候论证过程还没写出来，容易脱节。换成"先写依据、再据此定分类"，
    让 category 是在已经生成的 reasoning 文本基础上生成的，不需要多一次调用，
    只是调整字段顺序。
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
        "要求：先写 reasoning(一句话依据)，再基于这句依据给出 category(2-4字宏观主分类)，"
        "最后给出 tags(3-6个细粒度关键词)。\n"
        "严格输出纯JSON，不要markdown标记：\n"
        '{"reasoning": "...", "category": "...", "tags": ["...", "..."]}\n\n'
        "待分析的文件如下：\n"
        f"所在目录: {parent_dir}\n文件名: {filename}\n内容片段: {snippet}"
    )

    # temperature=0.1 不是 0，偶尔会吐出格式有瑕疵的 JSON（真实验证过：同样的 prompt、
    # 同样的参数，原样重新调用一次，大概率就正常了，不是内容本身有什么特殊之处触发的
    # 结构性问题）。给 2 次机会，比因为一次采样抖动就把这份文件打成 unclassified 更划算。
    last_error = None
    for attempt in range(2):
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
                raise ValueError("未找到 JSON")
            parsed = json.loads(content[start:end + 1])
            reasoning = str(parsed.get("reasoning", "")).strip()
            category = str(parsed.get("category", "")).strip()
            tags = [str(t).strip() for t in parsed.get("tags", []) if str(t).strip()]
            if not category:
                raise ValueError("category 为空")
            return {"category": category, "tags": tags, "reasoning": reasoning}
        except Exception as e:
            last_error = e
    print(f"  [Warning] 分类失败 {filename}（重试后仍失败）: {last_error}", flush=True)
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


def run_dynamic_classification_pipeline(
    index,
    embedder=None,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    min_content_length: int = MIN_CONTENT_LENGTH,
) -> dict:
    """V2 端到端逐文件分类管线，见模块顶部的分阶段说明。"""

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

    # Stage②：逐文件独立判断——每份文件都真实读取内容，互不参考彼此的判断。
    print(f"[Step 2] {max_workers} 路并发调用 SLM（{len(to_classify)} 份逐文件独立判断）...", flush=True)
    raw_results: Dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_classify_one_document, client, llm_model, row["path"], row["full_text"]): row["document_id"]
            for row in to_classify
        }
        done = 0
        for future in as_completed(futures):
            doc_id = futures[future]
            result = future.result()
            if result:
                raw_results[doc_id] = result
            done += 1
            if done % 50 == 0 or done == len(to_classify):
                print(f"  [Step 2] 已完成 {done}/{len(to_classify)}", flush=True)

    parse_failed = [row["document_id"] for row in to_classify if row["document_id"] not in raw_results]
    print(f"[Step 2] 分类完成：{len(raw_results)} 份成功，{len(parse_failed)} 份解析失败。", flush=True)

    # Stage③：同文件夹内标签统一。每份文件的 category 都是基于真实内容独立判断出来的，
    # 但同一个文件夹里的文件，判断结果可能只是措辞不同、说的是同一件事——按"直接父
    # 目录"分组（不管这个目录嵌套了几层，判断范围永远是文件和它的直接父目录这两级），
    # 复用 Stage④ taxonomy 归一化的同一套安全网（向量粗筛+LLM核实+代码按数量选标准名），
    # 只是把范围从"全库"缩小到"同一个文件夹内"——像才合并，文件夹本来就是散装的
    # 不会被强行合并。
    print("[Step 3] 同文件夹内标签统一...", flush=True)
    folder_groups: Dict[str, list] = {}
    for row in to_classify:
        if row["document_id"] in raw_results:
            folder_groups.setdefault(os.path.dirname(row["path"]), []).append(row["document_id"])

    folder_unify_count = 0
    for doc_ids in folder_groups.values():
        if len(doc_ids) < 2:
            continue
        folder_counts: Dict[str, int] = {}
        for doc_id in doc_ids:
            cat = raw_results[doc_id]["category"]
            folder_counts[cat] = folder_counts.get(cat, 0) + 1
        if len(folder_counts) < 2:
            continue  # 这个文件夹里已经是同一个分类了，不用处理
        canonical_map = _normalize_taxonomy(
            sorted(folder_counts.keys()), client, llm_model, embedder, category_counts=folder_counts
        )
        for doc_id in doc_ids:
            old_cat = raw_results[doc_id]["category"]
            new_cat = canonical_map.get(old_cat, old_cat)
            if new_cat != old_cat:
                raw_results[doc_id]["category"] = new_cat
                folder_unify_count += 1
    print(f"[Step 3] {folder_unify_count} 份文件的分类因跟同文件夹内其他文件合并而调整。", flush=True)

    # Stage④：Taxonomy 归一化（对象是分类名字符串，不涉及向量），这次是全库范围，
    # 在 Stage③ 已经做过一轮文件夹内合并的基础上，再合并一次跨文件夹的同义分类名。
    print("[Step 4] Taxonomy 归一化...", flush=True)
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
    print(f"[Step 4] {len(raw_categories)} 个原始分类名归一化为 {len(canonical_categories)} 个。", flush=True)

    # Stage⑤：置信度计算（category 和 tag 用同一套数学：文档向量分别跟候选文本的
    # 向量做点积）。这个分数只是存下来给人参考，不再用来推翻 Stage② 里 SLM 自己
    # 判定的分类——为什么不拿它去做"复核"，见上面模块顶部说明。
    print("[Step 5] 计算文档向量与分类名/标签向量的点积置信度...", flush=True)
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

    # Stage⑥：非文档类型 / 没有成功提取过文本的文件（图片、视频、代码、压缩包……），
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
        print(f"[Step 6] {len(basic_rows)} 份非文档类型/提取失败的文件，完成基础标签。", flush=True)

    # Stage⑦：持久化
    print("[Step 7] 持久化写入 SQLite...", flush=True)
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
