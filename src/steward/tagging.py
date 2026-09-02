"""端侧文档打标签引擎 —— 只做开放式内容理解，不再维护一套分类体系。

这是这周（V1 聚类打标签 → V2 开集自由分类 → V3 taxonomy 归纳+闭集分类）反复试错之后
做的决定：不再让每份文件收敛到一个"唯一正确"的分类。原因不是哪次没调好参数，是
真实数据反复证明这个目标本身和真实文件的性质冲突——文件天然多归属、边界模糊（一篇
"用工程师思维聊投资"的文章，本来就该同时挂"投资""工程师思维"这类标签，硬要它归到
唯一一个分类，答案会不稳定，snippet 覆盖不到的内容还会逼模型编一段站不住脚的理由
去凑一个错误答案）。V3 那套"先归纳 taxonomy、再让每个文件闭集选择"的机制想用自动化
校验去弥补这个天然的不稳定，实测下来：taxonomy 归纳、抽样校验、文件夹一致性覆盖，
不管怎么加固，总能在新的真实文件（尤其是没有文件夹结构、内容天然发散的目录，比如
~/Downloads 根目录）上炸出新的边界案例。完整的踩坑记录、每一条真实证据、V3 为什么
最终被砍掉，见 docs/dynamic_classification_architecture.md 的"V3 复盘"一节，这里
不重复，只保留最新、真正在用的实现。

真正保留下来的能力只有两个，都是这几周反复压力测试下来没被真实数据打脸过的：
1. tags —— 开放式、多标签、不互斥，每份文档都基于真实内容独立判断，不受任何固定
   列表限制。这是唯一一个即便在 reasoning 逻辑全崩的文件上依然可靠的输出。
2. reasoning —— 开放式一句话内容描述，不再是"为了给一个分类结果做依据"，就是单纯
   描述这份文件是什么，不需要牵强地把内容往一个分类名上靠。

图片/视频/压缩包等非文档类型、以及提取失败的文件，不进入下面这条 SLM 语义理解的
管线，只在最后补一道基础类型标签（见 Stage 基础标签），不是被忽略。代码项目（用
.git/package.json 等标记文件识别出的整个目录）单独走 Stage-project，一次调用判断
整个项目的性质，不逐个源码文件处理——原因见 scan.py 的 find_project_roots()。

管线：
  ① 内容量预筛 —— 文本过短/无信息量，直接判 untagged，不调用 SLM
  Stage-project：代码项目单独打标签，一次调用判断整个项目的性质（不逐个源码文件）
  Stage 打标签：逐文件真实读取内容独立调用一次 SLM，输出 reasoning（开放式一句话
     描述）+ tags（3-6个细粒度关键词，自由发挥）。JSON 字段顺序让 reasoning 排在
     tags 前面，先写依据再据此给标签，减轻两者脱节的概率。JSON 合法性靠
     response_format=json_schema 结构性保证，不靠"提示模型别用直引号"这种概率性
     手段——真实测过，文字提示只能降低概率，不能保证模型一定听话。另外单独加了一道
     "reasoning 有没有正常收尾"的校验（见 _reasoning_looks_complete）：真实语料里
     抓到过 reasoning 生成到一半意外断掉、但 JSON 语法依然合法解析成功的案例（统一
     断在正要引用标题/内容里具体短语的地方），不这么校验的话，这类残句会被当"成功"
     悄悄存进数据库。
  Stage 置信度 —— 文档向量分别跟每个 tag 的向量做点积，取平均值存下来当参考信息，
     不影响标签本身会不会被保存。
  Stage 基础标签 —— 图片/视频/压缩包等不进入上面语义理解的文件，用 Week 1 已经
     算好的 basic_type 补一道浅层标签
  Stage 持久化 —— 写入 SQLite
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


# Stage①：内容去空白后短于这个字符数，直接判 untagged，不调用 SLM
MIN_CONTENT_LENGTH = 30

# 并发调用 SLM 的线程数。这是推理引擎侧的配置，需要跟 llama-server 启动时的
# -np（slot 数）匹配，不要跟下面的打标签逻辑耦合。
DEFAULT_MAX_WORKERS = 8

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LLM_MODEL = "Qwen3.5-2B-Q4_K_M.gguf"


@dataclass
class TaggingResult:
    """打标签的标准输出契约（不再包含分类）。"""

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


# 判断 reasoning 是不是正常收尾的完整句子，不是生成中途断掉的残句。真实撞过的坑：
# response_format=json_schema 只保证 JSON 语法合法（字符串按时闭合），不保证字符串
# 内容本身是完整的——真实语料里抓到过十几份文件的 reasoning 直接断在"文件标题明确
# 提及"这种半句话上，JSON 依然合法解析成功，不会被"解析失败重试"这道校验发现，被
# 当"成功"悄悄存进数据库。这些案例统一有一个特征：全部断在正要引用标题/内容里具体
# 短语的地方，句子没写完、没有句末标点。用这个低成本、高准确的信号做兜底校验，
# 判定不完整就当失败处理，交给上层重试（不是瞎猜的阈值，是从真实坏案例里反推出来的）。
_SENTENCE_END_CHARS = ("。", "！", "？", "」", "》", ".", "!", "?")


def _reasoning_looks_complete(reasoning: str) -> bool:
    return reasoning.rstrip().endswith(_SENTENCE_END_CHARS)


def _tag_document(client: OpenAI, model_name: str, path: str, full_text: str) -> Optional[dict]:
    """给单个文件生成开放式的一句话内容描述（reasoning）和标签（tags）。返回
    {tags, reasoning}，失败返回 None。

    不再从固定分类列表里选（V3 的做法，已经废弃，原因见模块顶部说明）。tags 继续
    保持开放式、多标签、自由发挥——cowork 自己坦白过，它的 tags 大部分是靠文件名
    关键词匹配出来的，不是真读内容，质量因此完全依赖用户写文件名的习惯。我们不复刻
    这个短板：每个文件都真调用一次，基于真实内容判断。
    """

    filename = os.path.basename(path)
    parent_dir = os.path.basename(os.path.dirname(path))
    snippet = _extract_snippet(full_text, path)

    prompt = (
        "你是一个专业的端侧文件整理助手。请分析给定的文件，用一句话概括这份文件的核心内容"
        "（reasoning，不需要往某个分类上靠，如实描述即可），再给出 3-6 个细粒度关键词作为"
        "标签（tags）——标签之间不需要互斥，可以从不同角度描述同一份文件（比如一篇用工程师"
        "思维聊投资的文章，可以同时挂\"投资\"和\"工程师思维\"两个标签）。\n\n"
        "待分析的文件如下：\n"
        f"所在目录: {parent_dir}\n文件名: {filename}\n内容片段: {snippet}"
    )

    response_schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reasoning", "tags"],
    }

    last_error = None
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是严格只返回JSON格式的文件内容理解助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=200,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                response_format={"type": "json_schema", "json_schema": {"name": "tagging", "schema": response_schema}},
            )
            parsed = json.loads(resp.choices[0].message.content)
            reasoning = str(parsed.get("reasoning", "")).strip()
            tags = [str(t).strip() for t in parsed.get("tags", []) if str(t).strip()]
            if not reasoning:
                raise ValueError("reasoning 为空")
            if not _reasoning_looks_complete(reasoning):
                raise ValueError(f"reasoning 疑似生成中途断掉，没有正常收尾: {reasoning!r}")
            if not tags:
                raise ValueError("tags 为空")
            return {"tags": tags, "reasoning": reasoning}
        except Exception as e:
            last_error = e
    print(f"  [Warning] 打标签失败 {filename}（重试后仍失败）: {last_error}", flush=True)
    return None


def _tag_project(client: OpenAI, model_name: str, project_path: str, summary: str) -> Optional[dict]:
    """判断一个代码项目整体的性质（不是逐个源码文件判断）。输入只有项目根目录下的
    自描述内容（README/CLAUDE.md/package.json 摘要），不读项目内部的代码文件——
    这是验证过对代码项目更有效的信号来源，一个随手抽出来的源码文件单独看，
    看不出整个项目是干什么的，项目名字和 README 本身才是自然语言描述。

    这里的 category 字段本身就是开放式短语（不是从固定列表选），跟整体"不维护分类
    体系"的方向不冲突；调用方会把这个短语并进 tags 一起存，不再单独当分类字段处理。
    """
    project_name = os.path.basename(project_path.rstrip("/"))
    if not summary.strip():
        summary = "（没有找到 README/CLAUDE.md/package.json 等自描述文件，只能凭项目名字判断）"

    prompt = (
        "你是一个专业的端侧文件整理助手。请判断给定的代码项目整体属于什么性质。\n\n"
        "要求：先写 reasoning(一句话依据)，再基于这句依据给出 category(4-8字，比如"
        "\"个人技术项目\"、\"第三方开源仓库\"、\"学习练习项目\"、\"工作业务项目\"这类"
        "描述项目性质的短语，不是描述技术栈)，最后给出 tags(3-6个关键词，可以包含"
        "技术栈、项目用途等细节)。\n\n"
        f"项目名: {project_name}\n自描述内容:\n{summary[:2500]}"
    )

    response_schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "category": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reasoning", "category", "tags"],
    }

    last_error = None
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是严格只返回JSON格式的项目打标签助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=250,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                response_format={"type": "json_schema", "json_schema": {"name": "project_tagging", "schema": response_schema}},
            )
            parsed = json.loads(resp.choices[0].message.content)
            reasoning = str(parsed.get("reasoning", "")).strip()
            category = str(parsed.get("category", "")).strip()
            tags = [str(t).strip() for t in parsed.get("tags", []) if str(t).strip()]
            if not category:
                raise ValueError("category 为空")
            return {"category": category, "tags": tags, "reasoning": reasoning}
        except Exception as e:
            last_error = e
    print(f"  [Warning] 项目打标签失败 {project_name}（重试后仍失败）: {last_error}", flush=True)
    return None


def run_tagging_pipeline(
    index,
    embedder=None,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    min_content_length: int = MIN_CONTENT_LENGTH,
    force: bool = False,
) -> dict:
    """端到端打标签管线，见模块顶部的分阶段说明。

    force=False（默认）时走增量：一份文档如果已经有打标签结果、且它的内容
    自那以后没有被重新提取过（extractions.created_at 没有比 tagging_results.
    created_at 更新），就跳过，不重新调用 SLM——复用现成的两个时间戳，不需要
    新增字段。这只覆盖"内容变了要不要重新打"这一种情况，覆盖不了"打标签的
    逻辑本身变了"（改了 prompt、换了 snippet 长度、换了模型）——这种情况
    即使内容一个字都没变，也需要重新打，增量判断天生看不出"代码变了"，
    只能靠人自己决定要不要传 force=True 强制全部重来。
    """

    from steward.document_vectors import get_all_document_vectors
    from steward.embeddings import LocalEmbedder

    if embedder is None:
        embedder = LocalEmbedder()

    client = _build_llm_client(llm_base_url)

    # 以前这里 Step 0 会无条件先清空全部旧打标签数据（DELETE 整张表），再重新
    # 逐个处理——实测验证过这个顺序会在崩溃时把数据库拖进比运行前更差的状态：
    # 旧数据已经被删了，如果中途崩溃（比如 llama-server 挂了），还没处理到的
    # 文件会变成"完全没有标签"，不是"保留旧标签、等下次再刷新"。改成不再无
    # 条件清空——save_tagging_result() 本身对每个文档做的是 upsert（按
    # document_id 原地覆盖 + 标签列表先删后加），单个文档重新处理时天然会拿新
    # 结果覆盖旧结果，不需要一开始就把全库清空来保证"不会有旧数据残留"。
    # 代价：如果某份文档以前打过标签、这次运行没有再碰到它（目前 tag 命令还是
    # 全量处理所有 is_present=1 的文档，这个代价现在不会真的发生），它的旧标签
    # 会原样保留而不是被清空，这是可以接受的——"保留旧结果"永远好于"用户以为
    # 打过标签、实际上被清空后没能重新写入"。
    print("[Step 1] 加载已提取文本的文档...", flush=True)
    all_candidates_count = index.connection.execute(
        """
        SELECT COUNT(*) FROM documents d JOIN extractions x ON x.document_id = d.id
        WHERE d.is_present = 1 AND x.status = 'success' AND d.basic_type != 'project'
        """
    ).fetchone()[0]
    if force:
        rows = index.connection.execute(
            """
            SELECT d.id AS document_id, d.path AS path, x.full_text AS full_text
            FROM documents d JOIN extractions x ON x.document_id = d.id
            WHERE d.is_present = 1 AND x.status = 'success' AND d.basic_type != 'project'
            """
        ).fetchall()
    else:
        # LEFT JOIN tagging_results：没有打过标签的文档（tr.document_id IS NULL）
        # 一定要处理；打过标签的，只有内容比标签新（x.created_at > tr.created_at，
        # 也就是这份文档自打上标签之后又被重新提取过）才需要处理，否则跳过。
        rows = index.connection.execute(
            """
            SELECT d.id AS document_id, d.path AS path, x.full_text AS full_text
            FROM documents d
            JOIN extractions x ON x.document_id = d.id
            LEFT JOIN tagging_results tr ON tr.document_id = d.id
            WHERE d.is_present = 1 AND x.status = 'success' AND d.basic_type != 'project'
              AND (tr.document_id IS NULL OR x.created_at > tr.created_at)
            """
        ).fetchall()
    skipped_up_to_date = all_candidates_count - len(rows)
    print(
        f"[Step 1] 共 {all_candidates_count} 份文档，{skipped_up_to_date} 份已是最新跳过"
        f"（{'force 模式开启，不跳过' if force else '未开 force'}），{len(rows)} 份进入处理。",
        flush=True,
    )

    # Stage-project：代码项目单独打标签，数量天然很少（几个到几十个），逐个真实调用一次就够了。
    # 增量判断跟上面文档那部分是同一套逻辑，一并处理。
    if force:
        project_rows = index.connection.execute(
            """
            SELECT d.id AS document_id, d.path AS path, x.full_text AS full_text
            FROM documents d JOIN extractions x ON x.document_id = d.id
            WHERE d.is_present = 1 AND x.status = 'success' AND d.basic_type = 'project'
            """
        ).fetchall()
    else:
        project_rows = index.connection.execute(
            """
            SELECT d.id AS document_id, d.path AS path, x.full_text AS full_text
            FROM documents d
            JOIN extractions x ON x.document_id = d.id
            LEFT JOIN tagging_results tr ON tr.document_id = d.id
            WHERE d.is_present = 1 AND x.status = 'success' AND d.basic_type = 'project'
              AND (tr.document_id IS NULL OR x.created_at > tr.created_at)
            """
        ).fetchall()
    project_tagged_count = 0
    project_failed_count = 0
    if project_rows:
        print(f"[Step 1-project] {len(project_rows)} 个代码项目，逐个打标签...", flush=True)
        for row in project_rows:
            result = _tag_project(client, llm_model, row["path"], row["full_text"] or "")
            if result:
                # 项目性质（比如"个人技术项目"）本身就是个开放式短语，跟 tags 是同一种
                # 东西，直接并进 tags 列表，不再单独存一个"category"字段。
                tags = [(t, 1.0) for t in result["tags"]]
                if result["category"]:
                    tags.append((result["category"], 1.0))
                index.save_tagging_result(
                    document_id=row["document_id"],
                    tags=tags,
                    confidence=0.0,
                    status="tagged",
                    reasoning=result["reasoning"],
                )
                project_tagged_count += 1
            else:
                index.save_tagging_result(
                    document_id=row["document_id"],
                    tags=[],
                    confidence=0.0,
                    status="untagged",
                    reasoning="项目打标签 SLM 输出解析失败",
                )
                project_failed_count += 1

    # Stage①：内容量预筛
    to_tag = []
    skipped_short = []
    for row in rows:
        text = (row["full_text"] or "").strip()
        if len(text) < min_content_length:
            skipped_short.append(row["document_id"])
        else:
            to_tag.append(row)
    print(f"[Step 1] 内容过短跳过 {len(skipped_short)} 份，{len(to_tag)} 份进入 SLM 打标签。", flush=True)

    # Stage 打标签：逐文件真实读取内容独立调用一次 SLM，输出开放式 reasoning + tags。
    print(f"[Step 2] {max_workers} 路并发调用 SLM（{len(to_tag)} 份逐文件打标签）...", flush=True)
    raw_results: Dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_tag_document, client, llm_model, row["path"], row["full_text"]): row["document_id"]
            for row in to_tag
        }
        done = 0
        for future in as_completed(futures):
            doc_id = futures[future]
            result = future.result()
            if result:
                raw_results[doc_id] = result
            done += 1
            if done % 50 == 0 or done == len(to_tag):
                print(f"  [Step 2] 已完成 {done}/{len(to_tag)}", flush=True)

    parse_failed = [row["document_id"] for row in to_tag if row["document_id"] not in raw_results]
    print(f"[Step 2] 打标签完成：{len(raw_results)} 份成功，{len(parse_failed)} 份失败。", flush=True)

    # Stage 置信度：文档向量分别跟每个 tag 的向量做点积，取平均值存下来当参考信息——
    # 不再有分类体系，这个数字现在纯粹衡量"打出来的标签跟文档内容贴不贴"，不用来
    # 否决任何结果（不像 V3 时代试过的"复核"逻辑，那个已经被证明会误伤，撤掉了）。
    print("[Step 3] 计算文档向量与标签向量的点积置信度...", flush=True)
    doc_vectors_map = get_all_document_vectors(index)
    final_results: Dict[int, TaggingResult] = {}

    def _score_tags(raw_tags: List[str], doc_vec) -> List[Tuple[str, float]]:
        if doc_vec is None or not raw_tags:
            return [(t, 0.5) for t in raw_tags]
        tag_matrix = np.array(embedder.embed_documents(raw_tags), dtype=np.float32)
        sims = np.clip(tag_matrix @ doc_vec, 0.0, 1.0)
        return list(zip(raw_tags, sims.tolist()))

    for doc_id, r in raw_results.items():
        doc_vec = doc_vectors_map.get(doc_id)
        scored_tags = _score_tags(r["tags"], doc_vec)
        avg_confidence = float(np.mean([s for _, s in scored_tags])) if scored_tags else 0.0
        final_results[doc_id] = TaggingResult(
            tags=scored_tags, confidence=avg_confidence, status="tagged", reasoning=r["reasoning"],
        )

    # 内容过短 / SLM 解析失败的，统一判 untagged
    for doc_id in skipped_short:
        final_results[doc_id] = TaggingResult(
            tags=[], confidence=0.0, status="untagged", reasoning="文本内容过短，跳过 SLM 打标签",
        )
    for doc_id in parse_failed:
        final_results[doc_id] = TaggingResult(
            tags=[], confidence=0.0, status="untagged", reasoning="SLM 输出 JSON 解析失败",
        )

    # Stage 基础标签：图片/视频/压缩包等非文档类型、或提取失败的文件，不调用 SLM，
    # 直接用 Week 1 已经算好的 basic_type 补一道浅层标签，跟真正走过语义理解的结果
    # 用 status 区分开（"basic" vs "tagged"），讲清楚这批文件只做了类型识别。
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
        final_results[row["document_id"]] = TaggingResult(
            tags=[(basic_type, 1.0), (ext, 1.0)],
            confidence=0.0,
            status="basic",
            reasoning="非文档类型或未成功提取文本，仅做基础类型识别，未做语义内容分析",
        )
    if basic_rows:
        print(f"[Step 4] {len(basic_rows)} 份非文档类型/提取失败的文件，完成基础标签。", flush=True)

    # Stage 持久化——注意这里不再在外面包一层 with index.connection:。之前那样
    # 写会让人以为"整批要么全部提交、要么全部回滚"，实测验证过这个保证不成立：
    # save_tagging_result() 内部自己也开了一层事务，sqlite3 的 with 不支持真正
    # 的嵌套，内层退出就已经真正提交到磁盘了，外层这一层只是摆设。保留内层
    # 各自独立提交是对的——每个文档的写入本来就该是独立的原子单元，不需要也
    # 不应该假装它们是一整个不可分割的批次，见 document_index.py 里
    # save_tagging_result() 的注释。
    print("[Step 5] 持久化写入 SQLite...", flush=True)
    for doc_id, result in final_results.items():
        index.save_tagging_result(
            document_id=doc_id,
            tags=result.tags,
            confidence=result.confidence,
            status=result.status,
            reasoning=result.reasoning,
        )

    tagged_count = sum(1 for r in final_results.values() if r.status == "tagged")
    basic_count = sum(1 for r in final_results.values() if r.status == "basic")
    untagged_count = len(final_results) - tagged_count - basic_count

    print(
        f"🎉 完成！共 {len(final_results) + len(project_rows)} 份，"
        f"{tagged_count} 份深度打标签，{basic_count} 份基础类型识别，"
        f"{untagged_count} 份未打标签，{project_tagged_count} 个代码项目已打标签"
        f"（{project_failed_count} 个失败）。",
        flush=True,
    )

    return {
        "total_documents": len(rows),
        "skipped_up_to_date_count": skipped_up_to_date,
        "tagged_count": tagged_count,
        "basic_count": basic_count,
        "untagged_count": untagged_count,
        "skipped_short_count": len(skipped_short),
        "parse_failed_count": len(parse_failed),
        "project_count": len(project_rows),
        "project_tagged_count": project_tagged_count,
        "project_failed_count": project_failed_count,
    }
