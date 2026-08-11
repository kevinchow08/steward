"""端侧文档语义分类与打标签引擎。

根据文档文本与元数据提取主分类、多标签、置信度及推理依据。
严格遵循低置信度（< 0.70）自动回退为 unclassified 的兜底原则。
"""

from dataclasses import dataclass
from typing import List


# 默认置信度回退阈值
CONFIDENCE_THRESHOLD = 0.70


@dataclass
class ClassificationResult:
    """分类与打标签的标准输出契约。"""

    category: str
    tags: List[str]
    confidence: float
    status: str
    reasoning: str


def classify_document_text(text: str, file_path: str = "", basic_type: str = "document") -> ClassificationResult:
    """分析文档正文，输出结构化的分类、标签与置信度。

    当前第一版使用确定性的规则/关键词提取引擎，
    后续可无缝替换或接入本地轻量 SLM 大模型推理。
    """

    if not text or not text.strip():
        return ClassificationResult(
            category="unclassified",
            tags=[],
            confidence=0.0,
            status="unclassified",
            reasoning="文档没有提取到有效文字内容",
        )

    text_lower = text.lower()
    path_lower = file_path.lower()

    # 1. 尝试匹配定义好的业务主分类规则
    matched_category = None
    confidence = 0.0
    reasoning = ""
    extracted_tags = []

    # 规则 1：招聘简历类
    if any(k in text_lower or k in path_lower for k in ("简历", "resume", "cv", "求职", "工作经历", "项目经验")):
        matched_category = "招聘简历"
        confidence = 0.95
        reasoning = "正文中包含简历、工作经历或项目经验等典型词汇"
        extracted_tags.extend(["简历", "人力资源"])

    # 规则 2：销售与财务数据类
    elif any(k in text_lower or k in path_lower for k in ("销售数据", "财报", "报销", "发票", "收支", "资产配置", "投资", "bogleheads")):
        matched_category = "财务销售"
        confidence = 0.90
        reasoning = "正文中包含财报、报销、发票或投资配置等财务特征"
        extracted_tags.extend(["财务", "数据"])

    # 规则 3：代码与开发工具/技术文档
    elif any(k in text_lower or k in path_lower for k in ("agent", "prompt", "mcp", "python", "javascript", "架构", "api", "git")):
        matched_category = "技术文档"
        confidence = 0.88
        reasoning = "正文中包含 Agent、API、架构或编程语言相关词汇"
        extracted_tags.extend(["技术", "开发"])

    # 规则 4：工作周报/总结类
    elif any(k in text_lower or k in path_lower for k in ("周报", "月报", "总结", "weekly report", "工作计划")):
        matched_category = "工作周报"
        confidence = 0.85
        reasoning = "正文中包含周报、月报或工作总结特征"
        extracted_tags.extend(["工作", "总结"])

    # 规则 5：个人证件与合同类
    elif any(k in text_lower or k in path_lower for k in ("身份证", "护照", "合同", "协议", "身份证号")):
        matched_category = "证件合同"
        confidence = 0.92
        reasoning = "正文中包含合同、协议或证件信息"
        extracted_tags.extend(["合同", "重要证件"])

    # 2. 提取泛化标签（根据高频词/环境词补充）
    if "agent" in text_lower:
        extracted_tags.append("Agent")
    if "2026" in text_lower or "2026" in path_lower:
        extracted_tags.append("2026")
    if "python" in text_lower:
        extracted_tags.append("Python")
    if "上海" in text_lower or "shanghai" in path_lower:
        extracted_tags.append("上海")

    # 去重并排序标签
    unique_tags = sorted(list(set(extracted_tags)))

    # 3. 校验置信度阈值：低于 0.70 触发 unclassified 兜底
    if matched_category is None or confidence < CONFIDENCE_THRESHOLD:
        return ClassificationResult(
            category="unclassified",
            tags=unique_tags,
            confidence=round(confidence, 2),
            status="unclassified",
            reasoning="内容无法高置信度归入已知主分类",
        )

    return ClassificationResult(
        category=matched_category,
        tags=unique_tags,
        confidence=round(confidence, 2),
        status="classified",
        reasoning=reasoning,
    )


def run_dynamic_classification_pipeline(
    index,
    embedder=None,
    llm=None,
    min_cluster_size: int = 3,
    tag_threshold: float = 0.55,
) -> dict:
    """运行 Phase 3 端到端 100% 动态分类与多标签打标管线。

    过程涵盖：
    1. Step 3.1: 全量文档 Chunk 向量加权池化
    2. Step 3.2: HDBSCAN 无监督密度聚类
    3. Step 3.3: 抽样文本送端侧 SLM (Qwen3.5-2B) 提取 Category 与 Tag Pool
    4. Step 3.4: NumPy 2D 矩阵乘法全量点积打标 S = D · T^T
    5. Step 3.5: 将结果全量持久化落盘至 SQLite 四张数据表
    """
    from steward.clustering import cluster_document_vectors
    from steward.cluster_llm import LocalHttpClusterLLM, analyze_clusters_metadata
    from steward.document_vectors import get_all_document_vectors
    from steward.embeddings import LocalEmbedder
    from steward.tag_matcher import match_tags_for_documents

    if embedder is None:
        embedder = LocalEmbedder()
    if llm is None:
        llm = LocalHttpClusterLLM(base_url="http://localhost:8080/v1")

    # 1. 获取所有文档向量
    doc_vectors_map = get_all_document_vectors(index)
    if not doc_vectors_map:
        return {"total_documents": 0, "clusters": 0, "tagged_documents": 0}

    # 2. 无监督 HDBSCAN 聚类
    clusters, outliers = cluster_document_vectors(doc_vectors_map, min_cluster_size=min_cluster_size)

    # 3. 本地 SLM 理解簇主题，提炼主分类与 Tag Pool
    cluster_meta_map = analyze_clusters_metadata(clusters, index, llm=llm)

    # 4. 2D 矩阵点积分数匹配标签
    match_results = match_tags_for_documents(
        doc_vectors_map=doc_vectors_map,
        cluster_metadata_map=cluster_meta_map,
        clusters=clusters,
        embedder=embedder,
        similarity_threshold=tag_threshold,
    )

    # 5. 持久化落盘到 SQLite 表
    # 建立 cluster_id -> ClusterResult 的映射，方便计算质心相似度置信度
    cluster_obj_map = {c.cluster_id: c for c in clusters}
    doc_cluster_map = {}
    for c in clusters:
        for did in c.doc_ids:
            doc_cluster_map[did] = c.cluster_id

    saved_count = 0
    with index.connection:
        # 5.1 先保存算法解耦层 semantic_clusters 记录，生成数据库 cluster_db_id
        cluster_db_id_map = {}
        for c in clusters:
            meta = cluster_meta_map.get(c.cluster_id)
            if meta:
                cat_id = index._get_or_create_category(meta.category)
                db_cid = index.save_semantic_cluster(
                    category_id=cat_id,
                    centroid_vector=c.centroid_vector,
                    tag_pool=meta.tag_pool,
                    summary=f"Cluster {c.cluster_id} contains {len(c.doc_ids)} documents",
                    confidence=0.90,
                )
                cluster_db_id_map[c.cluster_id] = db_cid

        # 5.2 保存文档级分类、标签绑定与数学点积置信度
        for did, match in match_results.items():
            cid = doc_cluster_map.get(did)
            cluster_obj = cluster_obj_map.get(cid) if cid is not None else None
            db_cid = cluster_db_id_map.get(cid) if cid is not None else None

            # 数学公式算置信度：文档向量与所属簇质心向量的点积余弦相似度
            if cluster_obj is not None and did in doc_vectors_map:
                doc_vec = doc_vectors_map[did]
                confidence = float(np.dot(doc_vec, cluster_obj.centroid_vector))
                confidence = max(0.0, min(1.0, confidence))  # 截断在 0.0 ~ 1.0 之间
            else:
                confidence = 0.50

            # 模板化拼装零 LLM 消耗 Reasoning 推理理由
            tags_str = ", ".join([f"{t[0]}({t[1]:.2f})" for t in match.matched_tags]) if match.matched_tags else "无"
            reasoning = f"HDBSCAN 向量空间自动分簇 (簇 #{cid})。质心相似度: {confidence:.2f}。匹配标签: {tags_str}。"

            # 保存主分类、数据库 cluster_id、数学置信度与模板 Reasoning
            index.save_classification(
                document_id=did,
                category_name=match.category,
                tags=match.matched_tags,
                confidence=confidence,
                status="classified" if match.category != "未分类" else "unclassified",
                reasoning=reasoning,
                cluster_id=db_cid,
            )

            saved_count += 1

    return {
        "total_documents": len(doc_vectors_map),
        "clusters": len(clusters),
        "outliers": len(outliers),
        "tagged_documents": saved_count,
    }

