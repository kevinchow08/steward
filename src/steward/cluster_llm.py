"""端侧小模型 (SLM) Cluster 命名与 Tag Pool 提炼模块

支持通过本地 HTTP API (llama.cpp server / Ollama) 或离线启发式规则降级，
对 HDBSCAN 产生的聚类簇样本进行语义理解，生成主分类名称 (Category) 与候选标签池 (Tag Pool)。
"""

import json
from dataclasses import dataclass
from typing import List

import httpx
from openai import OpenAI


@dataclass
class ClusterMetadata:
    """单个语义簇被 LLM 理解后的元数据。"""

    category: str
    tag_pool: List[str]


class BaseClusterLLM:
    """端侧 SLM 推理抽象基类。"""

    def generate_cluster_metadata(self, representative_texts: List, existing_categories: List[str] = None) -> ClusterMetadata:
        raise NotImplementedError


class LocalHttpClusterLLM(BaseClusterLLM):
    """使用标准 OpenAI SDK 访问本地 SLM 服务 (llama.cpp server / Ollama / vLLM)。"""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model_name: str = "Qwen3.5-2B-Q4_K_M.gguf",
    ):
        # 显式使用 trust_env=False 禁用代理对 localhost 8080 端口的误拦截 (彻底解决 502 Bad Gateway)
        # timeout=30.0 防止 LLM 服务无响应时无限阻塞挂死
        self.client = OpenAI(
            base_url=base_url,
            api_key="no-key-required",
            http_client=httpx.Client(trust_env=False, timeout=30.0),
        )
        self.model_name = model_name

    def generate_cluster_metadata(
        self,
        representative_texts: List,   # List of (full_text, file_path) tuples
        existing_categories: List[str] = None,
    ) -> ClusterMetadata:
        def _extract_snippet(text: str, file_path: str = "", max_len: int = 300) -> str:
            """提取文档的有效正文片段，根据文件类型做差异化处理。

            策略：
            - HTML 文件：先剥离 HTML 标签，再跳过前 200 字符的 <head>/导航区
            - PDF  文件：不含 HTML 标签，但发票/表单 PDF 的前段是固定表头字段
                         （如"购买方信息 统一社会信用代码..."），跳过更多（300字）取实质内容
            - 其他文件（.md/.txt）：直接从头取，内容通常从第一行就有语义
            """
            import re
            ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""

            # Step 1: 剥离 HTML 标签（仅对 HTML 文件有效，PDF/txt 不含 HTML 标签，re.sub 对其无副作用）
            clean = re.sub(r"<[^>]+>", " ", text)
            # Step 2: 合并多余空白
            clean = re.sub(r"\s+", " ", clean).strip()

            if ext in ("html", "htm"):
                # HTML 文件：前 200 字符通常是 <head>/<meta>/导航链接等无意义内容
                start = 200 if len(clean) > 400 else 0
            elif ext == "pdf":
                # PDF 文件（尤其是发票/表单）：前段是固定表头字段名，语义模板化，无助于 SLM 分类
                # 例如："电子发票 购买方信息 统一社会信用代码 名称 项目名称 规格型号..."
                # 跳过前 350 字符，取实质性正文内容（金额数字、商品名称、备注等）
                start = 350 if len(clean) > 600 else 0
            else:
                # .md / .txt 等纯文本：内容从第一行就有语义，直接从头取
                start = 0

            return clean[start : start + max_len].strip()

        import os
        docs_summary_parts = []
        for i, (text, path) in enumerate(representative_texts):
            # 文件名和父目录是最重要的上下文信号
            # 例如：/doc/Nest 通关秘籍/109.会议室预订系统.md
            # →「Nest 通关秘籍」说明这是技术课程，「会议室预订系统」只是课程里的实战项目名，不是分类依据
            filename = os.path.basename(path) if path else "未知文件"
            parent_dir = os.path.basename(os.path.dirname(path)) if path else ""
            snippet = _extract_snippet(text, path)
            docs_summary_parts.append(
                f"- 文件 {i+1}\n"
                f"  所在目录: {parent_dir}\n"
                f"  文件名: {filename}\n"
                f"  内容片段: {snippet}"
            )
        docs_summary = "\n".join(docs_summary_parts)

        existing_info = ""
        if existing_categories:
            cat_list_str = "、".join(set(existing_categories))
            existing_info = (
                f"\n【强制约束】当前已存在的主分类池：[{cat_list_str}]。\n"
                "你必须优先从已有分类中选择。只要语义接近（例如都是发票/报销/财务相关），就必须直接复用已有名称，禁止新建语义重叠的分类！\n"
                "仅当该簇内容属于完全不同的全新领域，且已有分类中没有任何语义相近选项时，才允许新建分类名称。\n"
            )

        prompt = (
            "你是一个专业的端侧文件整理 Agent。以下是一组属于同一个主题的代表性文件：\n\n"
            f"{docs_summary}\n\n"
            f"{existing_info}\n"
            "分类约束说明：\n"
            "1. 【最重要】优先根据'所在目录'和'文件名'来判断文件的用途类型，而不是被内容里的具体业务词汇误导。\n"
            "   例如：目录是'Nest 通关秘籍'的文件，无论内容涉及'会议室'还是'聊天室'，本质都是技术教程，分类应为技术代码/后端开发。\n"
            "   例如：目录是'报销相关'的文件，无论文件名是什么，分类应为财务报销。\n"
            "2. category 必须是高度抽象的顶级宏观分类 (2-4个字，如: 财务报销 / 技术代码 / 证件合同 / 运动健身 / 投资理财)。\n"
            "3. 严禁把具体业务词（如: 会议室、聊天室、餐饮、发票）填入 category！这些是 tag_pool 的内容。\n\n"
            "请严格输出如下格式的纯 JSON，绝对不要包含任何 Markdown 标记或多余解释：\n"
            "{\n"
            '  "category": "极简的高层主分类名称",\n'
            '  "tag_pool": ["提取5-8个最核心的细粒度候选关键词标签"]\n'
            "}"
        )

        try:
            # 标准 OpenAI SDK 发起调用 (通过 extra_body 强制禁用思维链，实现 0.1s 极速 JSON 直出)
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个严格仅返回 JSON 格式的端侧分类助手。直接输出 JSON 字符串，禁止输出 Markdown 标记。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = resp.choices[0].message.content.strip()

            # 最外层 JSON 提取，抗干扰
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                content = content[start_idx : end_idx + 1]

            parsed = json.loads(content)
            category = parsed.get("category", "未分类").strip()
            tag_pool = [t.strip() for t in parsed.get("tag_pool", []) if t.strip()]

            return ClusterMetadata(category=category, tag_pool=tag_pool)

        except Exception as e:
            print(f"[Warning] 本地 LLM 推理未就绪或响应失败 ({e})，触发启发式规则降级。")
            fallback = HeuristicFallbackClusterLLM()
            return fallback.generate_cluster_metadata(representative_texts)


class HeuristicFallbackClusterLLM(BaseClusterLLM):
    """在无本地 LLM 服务运行时的启发式规则降级分析器 (保证 Pipeline 绝对不崩溃)。"""

    def generate_cluster_metadata(self, representative_texts: List, existing_categories: List[str] = None) -> ClusterMetadata:
        # 兼容 (text, path) 元组列表或纯字符串列表
        combined = " ".join([t[0] if isinstance(t, tuple) else t for t in representative_texts]).lower()

        if any(k in combined for k in ("发票", "报销", "餐饮", "交通", "费用", "收据", "金额")):
            return ClusterMetadata(
                category="财务报销",
                tag_pool=["发票", "报销", "餐饮", "交通", "费用", "消费", "凭证"],
            )

        if any(k in combined for k in ("简历", "resume", "cv", "工作经历", "项目经验")):
            return ClusterMetadata(
                category="招聘简历",
                tag_pool=["简历", "工作经历", "项目经验", "教育背景", "个人技能"],
            )

        if any(k in combined for k in ("agent", "python", "mcp", "api", "git", "javascript", "架构")):
            return ClusterMetadata(
                category="技术代码",
                tag_pool=["Agent", "Python", "架构", "API", "代码", "开发"],
            )

        if any(k in combined for k in ("周报", "月报", "总结", "weekly", "工作计划")):
            return ClusterMetadata(
                category="工作周报",
                tag_pool=["周报", "总结", "工作计划", "项目进展"],
            )

        return ClusterMetadata(
            category="综合文档",
            tag_pool=["文档", "资料", "笔记", "参考"],
        )


def analyze_clusters_metadata(
    clusters: list,
    index,
    llm: BaseClusterLLM = None,
) -> dict:
    """遍历所有 HDBSCAN 主题簇，提取代表性文件内容，调用 SLM 产出簇级元数据。

    流程：
    1. 对每个簇调用 SLM 产出初步的 category + tag_pool
    2. 最后做一次「Taxonomy 归一化」：把语义重叠的分类名合并为统一名称
       （解决"前端开发" vs "技术代码"这类顺序依赖产生的同义词分裂问题）

    :return: 字典 {cluster_id: ClusterMetadata}
    """

    if llm is None:
        # 默认尝试连接 llama.cpp server (端口 8080)；失败会自动触发 HeuristicFallback
        llm = LocalHttpClusterLLM(base_url="http://localhost:8080/v1")

    cluster_metadata_map = {}
    known_categories: List[str] = []
    total = len(clusters)

    for idx, c in enumerate(clusters):
        print(f"  [Step 3.3] 分析 Cluster #{c.cluster_id} ({idx+1}/{total}, {len(c.doc_ids)} 份文档)...", flush=True)
        rep_texts = []
        for did in c.representative_doc_ids:
            row = index.connection.execute(
                """
                SELECT e.full_text, d.path
                FROM extractions e
                JOIN documents d ON d.id = e.document_id
                WHERE e.document_id = ?
                """,
                (did,),
            ).fetchone()
            if row and row["full_text"]:
                # 传入 (full_text, file_path) 二元组，让 _extract_snippet 按文件类型选择策略
                rep_texts.append((row["full_text"], row["path"]))

        if not rep_texts:
            rep_texts = [("空文档", "")]


        # 2. 调用 SLM 生成该簇的 Category 与 Tag Pool (传入已有的主分类上下文，保证收敛)
        meta = llm.generate_cluster_metadata(rep_texts, existing_categories=known_categories)
        if meta and meta.category and meta.category != "未分类":
            known_categories.append(meta.category)

        cluster_metadata_map[c.cluster_id] = meta

    # === Taxonomy 归一化 Pass ===
    # 问题：SLM 独立命名每个簇，顺序依赖导致语义相近的簇产生同义分类名
    # 例如："前端开发"和"技术代码"语义重叠，但因为处理顺序不同而被创建为两个独立分类
    # 修复：收集所有唯一分类名，让 SLM 做一次最终的合并归一，建立统一 taxonomy
    all_categories = list({m.category for m in cluster_metadata_map.values() if m and m.category})
    if len(all_categories) > 1:
        print(f"  [Step 3.3] Taxonomy 归一化：合并 {len(all_categories)} 个分类...", flush=True)
        canonical_map = _normalize_taxonomy(all_categories, llm)
        # 将所有簇的 category 映射到归一化名称
        for cid, meta in cluster_metadata_map.items():
            if meta and meta.category in canonical_map:
                cluster_metadata_map[cid] = ClusterMetadata(
                    category=canonical_map[meta.category],
                    tag_pool=meta.tag_pool,
                )

    return cluster_metadata_map


def _normalize_taxonomy(categories: List[str], llm: BaseClusterLLM) -> dict:
    """将一组粗粒度分类名归一化为统一 Taxonomy，合并语义重叠的名称。

    :param categories: 所有簇产出的原始分类名列表
    :param llm: SLM 实例
    :return: {原始分类名: 归一化后的标准分类名}
    """
    cat_list_str = "\n".join([f"- {c}" for c in categories])
    prompt = (
        "以下是一组文件分类名称，其中可能存在语义重叠或同义词（如'前端开发'与'技术代码'实质相同）：\n\n"
        f"{cat_list_str}\n\n"
        "请将语义相近的名称合并为同一个标准名称，建立一个精简、无重叠的顶级分类 Taxonomy。\n"
        "规则：\n"
        "1. 合并语义重叠的分类（如: 技术代码 / 前端开发 / 技术文档 → 统一为「技术代码」）\n"
        "2. 保留语义明确不同的分类（如: 财务报销 / 证件合同 / 英语学习 不应合并）\n"
        "3. 输出一个 JSON 对象，key 为原始名称，value 为归一化后的标准名称\n\n"
        "请严格输出纯 JSON，不要包含任何 Markdown 标记或解释：\n"
        "{\n"
        '  "原始名称1": "标准名称",\n'
        '  "原始名称2": "标准名称"\n'
        "}"
    )

    import json
    try:
        resp = llm.client.chat.completions.create(
            model=llm.model_name,
            messages=[
                {"role": "system", "content": "你是一个严格仅返回 JSON 的文件分类归一化助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = resp.choices[0].message.content.strip()
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            mapping = json.loads(content[start:end+1])
            # 验证：确保只接受字符串值，过滤异常输出
            return {k: str(v) for k, v in mapping.items() if isinstance(v, str)}
    except Exception as e:
        print(f"  [Warning] Taxonomy 归一化失败 ({e})，跳过合并。", flush=True)

    # 降级：原样返回（每个名称映射到自身）
    return {c: c for c in categories}

