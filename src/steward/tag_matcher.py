"""NumPy 矩阵点积全量打标引擎 (Tag Matcher)

利用 2D 矩阵乘法 S = D · T^T，一次性计算全量文档向量与所有候选 Tag 向量的高维余弦相似度，
实现 0 LLM 算力消耗、毫秒级的多标签 (Multi-Tag) 精准匹配与置信度过滤。
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class DocumentMatchResult:
    """单份文档被矩阵打标后的匹配结果。"""

    doc_id: int
    category: str
    matched_tags: List[Tuple[str, float]]  # [(tag_name, similarity_score)]


def match_tags_for_documents(
    doc_vectors_map: Dict[int, np.ndarray],
    cluster_metadata_map: dict,  # {cluster_id: ClusterMetadata}
    clusters: list,  # HDBSCAN Cluster 列表
    embedder,
    similarity_threshold: float = 0.55,
) -> Dict[int, DocumentMatchResult]:
    """使用 2D 矩阵乘法做全量文档对 Tag 标签池的点积相似度匹配。

    :param doc_vectors_map: {doc_id: 1024维 np.ndarray}
    :param cluster_metadata_map: {cluster_id: ClusterMetadata(category, tag_pool)}
    :param clusters: HDBSCAN ClusterResult 实例列表
    :param embedder: BGE-M3 向量化提取器 (LocalEmbedder)
    :param similarity_threshold: 标签匹配置信度阈值 (默认 0.55)
    :return: {doc_id: DocumentMatchResult}
    """
    # 1. 建立 doc_id 到 cluster_id 的反向查找映射
    doc_cluster_map = {}
    for c in clusters:
        for did in c.doc_ids:
            doc_cluster_map[did] = c.cluster_id

    # 2. 收集所有簇的标签，生成全局去重 Tag 列表
    all_candidate_tags = set()
    for meta in cluster_metadata_map.values():
        if meta and meta.tag_pool:
            all_candidate_tags.update(meta.tag_pool)

    tag_list = sorted(list(all_candidate_tags))
    if not tag_list or not doc_vectors_map:
        return {}

    # 3. 使用 BGE-M3 将去重 Tag 批量向量化，构建 Tag 矩阵 T (M, 1024)
    tag_vectors = embedder.embed_documents(tag_list)  # np.ndarray (M, 1024)
    tag_matrix = np.array(tag_vectors, dtype=np.float32)

    # 4. 构建文档矩阵 D (N, 1024)
    doc_ids = list(doc_vectors_map.keys())
    doc_matrix = np.array([doc_vectors_map[did] for did in doc_ids], dtype=np.float32)  # (N, 1024)

    # 5. 执行模态 C 矩阵点积相乘 S = D · T^T (N, 1024) · (1024, M) = (N, M)
    similarity_matrix = np.dot(doc_matrix, tag_matrix.T)  # (N, M) 2D 相似度打分矩阵

    # 6. 遍历相似度矩阵，提取每份文档高于阈值的匹配 Tag
    results = {}
    for row_idx, did in enumerate(doc_ids):
        cid = doc_cluster_map.get(did)
        meta = cluster_metadata_map.get(cid) if cid is not None else None

        category = meta.category if meta else "未分类"

        # 提取当前文档对所有 Tag 的打分行 (M,)
        scores = similarity_matrix[row_idx]

        matched_tags = []
        for tag_idx, score in enumerate(scores):
            if score >= similarity_threshold:
                matched_tags.append((tag_list[tag_idx], float(score)))

        # 按相似度得分从高到低排序
        matched_tags.sort(key=lambda x: x[1], reverse=True)

        results[did] = DocumentMatchResult(
            doc_id=did,
            category=category,
            matched_tags=matched_tags,
        )

    return results
