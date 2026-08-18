"""端侧文档分类引擎 V2—— 逐文件直接分类。

每份文档独立调用一次本地 SLM，产出 category + tags + reasoning，不再依赖聚类簇的
集体判断（V1 的 Cluster-then-Label 设计已废弃，为什么废弃、V2 具体怎么设计，
见 docs/dynamic_classification_architecture.md，这里只放实现）。

六阶段管线：
  ① 内容量预筛 —— 文本过短/无信息量，直接判 unclassified，不调用 SLM
  ② 逐文件 SLM 分类 —— 并发调用，独立产出 category + tags + reasoning
  ③ Taxonomy 归一化 —— 合并同义分类名（对象是分类名字符串，不涉及向量）
  ④ 置信度计算与复核 —— 文档向量跟全部候选分类名做点积，检验 SLM 判定的分类
     是不是候选里排名最靠前（或跟最高分差距在容差内）的那个，不是就打回 unclassified
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

# Stage④：SLM 判定的分类，点积只要跟候选池里最高分差距不超过这个值就仍然采信。
# 这是排名容差，不是绝对阈值——不同分类名天生的点积基线不一样（越具体的分类名基线越高），
# 拿绝对值卡所有分类不合理，实测过（见架构文档）。这个容差数值还没有足够真实边缘案例校准，
# 是保守起步值，后续要跟着实际跑出来的数据回头调整。
CONFIDENCE_MARGIN = 0.03

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
    """调用一次 SLM，独立判断单份文档的分类。返回 {category, tags, reasoning}，解析失败返回 None。"""

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


def _normalize_taxonomy(categories: List[str], client: OpenAI, model_name: str) -> Dict[str, str]:
    """把一组分类名里语义重叠的合并为统一名称。

    :return: {原始分类名: 归一化后的标准分类名}，每个原始名都保证有映射（找不到就映射到自己）
    """
    if not categories:
        return {}
    if len(categories) == 1:
        return {categories[0]: categories[0]}

    cat_list_str = "\n".join([f"- {c}" for c in categories])
    prompt = (
        "以下是一组文件分类名称，其中可能存在语义重叠或同义词（如'前端开发'与'技术文档'实质相同）：\n\n"
        f"{cat_list_str}\n\n"
        "请将语义相近的名称合并为同一个标准名称，建立一个精简、无重叠的顶级分类 Taxonomy。\n"
        "规则：\n"
        "1. 合并语义重叠的分类（如: 技术代码 / 前端开发 / 技术文档 → 统一为「技术文档」）\n"
        "2. 保留语义明确不同的分类（如: 财务报销 / 证件合同 / 英语学习 不应合并）\n"
        "3. 输出一个 JSON 对象，key 为原始名称，value 为归一化后的标准名称，每个原始名称都要出现\n\n"
        "请严格输出纯 JSON，不要包含任何 Markdown 标记或解释：\n"
        '{"原始名称1": "标准名称", "原始名称2": "标准名称"}'
    )

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是严格只返回 JSON 的文件分类归一化助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=len(categories) * 40 + 200,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = resp.choices[0].message.content.strip()
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            mapping = json.loads(content[start:end + 1])
            result = {k: v for k, v in mapping.items() if isinstance(v, str)}
            # LLM 的回答不一定覆盖了每一个原始分类名（哪怕 prompt 里明确要求了，2B 小模型
            # 也可能漏掉几个）。setdefault(c, c)：result 里已经有 c 就不动，没有就补一条
            # "自己映射到自己"（代表这个分类名不用合并，原样保留）。这不是可有可无的装饰——
            # 后面算置信度排名的候选池就是取这个映射的 values，漏了就会在候选池里也漏掉，
            # 变成一个不容易发现的正确性问题。
            for c in categories:
                result.setdefault(c, c)
            return result
    except Exception as e:
        print(f"  [Warning] Taxonomy 归一化失败 ({e})，跳过合并，保留原始分类名。", flush=True)

    return {c: c for c in categories}


def run_dynamic_classification_pipeline(
    index,
    embedder=None,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    min_content_length: int = MIN_CONTENT_LENGTH,
    confidence_margin: float = CONFIDENCE_MARGIN,
) -> dict:
    """V2 端到端逐文件分类管线，见模块顶部的五阶段说明。"""

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

    # Stage②：逐文件并发 SLM 分类
    print(f"[Step 2] {max_workers} 路并发调用 SLM...", flush=True)
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

    # Stage③：Taxonomy 归一化（对象是分类名字符串，不涉及向量）
    print("[Step 3] Taxonomy 归一化...", flush=True)
    raw_categories = sorted({r["category"] for r in raw_results.values()})
    canonical_map = _normalize_taxonomy(raw_categories, client, llm_model)
    for r in raw_results.values():
        r["category"] = canonical_map.get(r["category"], r["category"])
    canonical_categories = sorted(set(canonical_map.values()))
    print(f"[Step 3] {len(raw_categories)} 个原始分类名归一化为 {len(canonical_categories)} 个。", flush=True)

    # Stage④：置信度计算与复核（借用③清理好的候选池，是文档级别的独立校验）
    # category 和 tag 的置信度用的是同一套数学：文档向量分别跟候选文本的向量做点积。
    # category 的候选池是归一化后的分类名列表；tag 没有归一化，候选池就是这份文档自己
    # 被打上的那几个标签——只需要知道"这个标签跟这篇文档本身贴合到什么程度"，不需要
    # 跟别的文档的标签比较，所以不用像 category 那样做跨文档的排名判断。
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

            sims = np.clip(np.dot(category_matrix, doc_vec), 0.0, 1.0)
            best_idx = int(np.argmax(sims))
            assigned_idx = category_index[assigned_category]
            assigned_sim = float(sims[assigned_idx])
            best_sim = float(sims[best_idx])

            if best_idx == assigned_idx or (best_sim - assigned_sim) <= confidence_margin:
                final_results[doc_id] = ClassificationResult(
                    category=assigned_category, tags=scored_tags, confidence=assigned_sim,
                    status="classified", reasoning=r["reasoning"],
                )
            else:
                final_results[doc_id] = ClassificationResult(
                    category="unclassified", tags=scored_tags, confidence=assigned_sim,
                    status="unclassified",
                    reasoning=(
                        f"SLM 判定为「{assigned_category}」(点积 {assigned_sim:.2f})，"
                        f"但「{canonical_categories[best_idx]}」点积更高 ({best_sim:.2f})，"
                        "差距超过容差，判定不可信。"
                    ),
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
    unclassified_count = len(final_results) - classified_count

    print(
        f"🎉 分类完成！共 {len(final_results)} 份，{classified_count} 份已分类，"
        f"{unclassified_count} 份未归类。",
        flush=True,
    )

    return {
        "total_documents": len(rows),
        "classified_count": classified_count,
        "unclassified_count": unclassified_count,
        "skipped_short_count": len(skipped_short),
        "parse_failed_count": len(parse_failed),
        "canonical_categories": len(canonical_categories),
    }
