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

    def generate_cluster_metadata(self, representative_texts: List[str]) -> ClusterMetadata:
        raise NotImplementedError


class LocalHttpClusterLLM(BaseClusterLLM):
    """使用标准 OpenAI SDK 访问本地 SLM 服务 (llama.cpp server / Ollama / vLLM)。"""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model_name: str = "Qwen3.5-2B-Q4_K_M.gguf",
    ):
        # 显式使用 trust_env=False 禁用代理对 localhost 8080 端口的误拦截 (彻底解决 502 Bad Gateway)
        self.client = OpenAI(
            base_url=base_url,
            api_key="no-key-required",
            http_client=httpx.Client(trust_env=False),
        )
        self.model_name = model_name

    def generate_cluster_metadata(
        self,
        representative_texts: List[str],
        existing_categories: List[str] = None,
    ) -> ClusterMetadata:
        docs_summary = "\n".join([f"- 文件 {i+1} 内容片段: {text[:250]}" for i, text in enumerate(representative_texts)])

        existing_info = ""
        if existing_categories:
            cat_list_str = "、".join(set(existing_categories))
            existing_info = f"\n当前已创建的主分类池：[{cat_list_str}]。请优先评估是否可以归入已有主分类；若语义匹配，请直接复用已有名称；仅当确属全新领域时才允许新建名称。\n"

        prompt = (
            "你是一个专业的端侧文件整理 Agent。以下是一组属于同一个主题的代表性文件内容片段：\n\n"
            f"{docs_summary}\n\n"
            f"{existing_info}\n"
            "分类约束说明：\n"
            "1. category 必须是高度抽象的顶级宏观分类 (2-4个字，如: 财务报销 / 招聘简历 / 技术代码 / 会议总结 / 证件合同)。\n"
            "2. 严禁把具体细节词 (如: 打车、餐饮、发票、Python) 填入 category！所有具体细节词必须全部归入 tag_pool。\n\n"
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

    def generate_cluster_metadata(self, representative_texts: List[str]) -> ClusterMetadata:
        combined = " ".join(representative_texts).lower()

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

    :return: 字典 {cluster_id: ClusterMetadata}
    """

    if llm is None:
        # 默认尝试连接 llama.cpp server (端口 8080)；失败会自动触发 HeuristicFallback
        llm = LocalHttpClusterLLM(base_url="http://localhost:8080/v1")

    cluster_metadata_map = {}
    known_categories: List[str] = []

    for c in clusters:
        # 1. 从代表性文档中读取正文片段
        rep_texts = []
        for did in c.representative_doc_ids:
            row = index.connection.execute(
                """
                SELECT full_text FROM extractions WHERE document_id = ?
                """,
                (did,),
            ).fetchone()
            if row and row["full_text"]:
                rep_texts.append(row["full_text"])

        if not rep_texts:
            rep_texts = ["空文档"]

        # 2. 调用 SLM 生成该簇的 Category 与 Tag Pool (传入已有的主分类上下文，保证收敛)
        meta = llm.generate_cluster_metadata(rep_texts, existing_categories=known_categories)
        if meta and meta.category and meta.category != "未分类":
            known_categories.append(meta.category)

        cluster_metadata_map[c.cluster_id] = meta

    return cluster_metadata_map
