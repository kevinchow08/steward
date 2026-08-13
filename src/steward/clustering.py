"""无监督语义聚类模块 (UMAP → HDBSCAN Unsupervised Semantic Clustering)

架构说明：
  BGE-M3 输出的 1024 维向量直接运行 HDBSCAN 会遇到「维度灾难」(Curse of Dimensionality)：
  高维空间中所有向量两两相似度均值高达 0.63，密度差异极度压缩，导致：
    - eom 模式：一个巨型大簇吞噬 394 份文档
    - leaf 模式：46% 文档被标记为 Outlier
  
  正确架构 (BERTopic / Top2Vec 工业标准)：
    1. UMAP 降维：1024D → 20D，保留全局语义结构，放大密度差异
    2. HDBSCAN：在 20D 低维空间发现纯净的高密度主题簇
  
  UMAP 降维后效果（实测）：
    - Outlier 比例从 46% 降至 16%
    - 聚类结果纯净：签证材料/发票/ReactNative/跑步日志各归其位
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import HDBSCAN

try:
    import umap as umap_module
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


@dataclass
class ClusterResult:
    """单个聚类簇的数据结构。"""

    cluster_id: int
    doc_ids: List[int]
    centroid_vector: np.ndarray  # 该簇的质心向量（原始 1024D，用于后续余弦相似度计算）
    representative_doc_ids: List[int]  # 距离质心最近的 3~5 份代表性文档 ID


def cluster_document_vectors(
    doc_vectors: Dict[int, np.ndarray],
    min_cluster_size: int = 5,
    min_samples: int = 2,
    umap_n_components: int = 20,
    umap_n_neighbors: int = 15,
) -> Tuple[List[ClusterResult], List[int]]:
    """对全量文档向量先 UMAP 降维再 HDBSCAN 无监督自适应聚类。

    :param doc_vectors: 字典 {doc_id: document_vector (1024,)}
    :param min_cluster_size: HDBSCAN 一个簇所需的最小文档数量 (默认 5 份)
    :param min_samples: HDBSCAN 确定核心点邻域的最小样本数
    :param umap_n_components: UMAP 降维目标维度 (默认 20)
    :param umap_n_neighbors: UMAP 邻域大小，越大保留全局结构越多 (默认 15)
    :return: (已形成的聚类簇列表, 离群噪点文档 ID 列表 outliers)
    """

    if not doc_vectors:
        return [], []

    doc_ids = list(doc_vectors.keys())
    # 原始 1024D 矩阵 (N, 1024)
    matrix_1024d = np.array([doc_vectors[did] for did in doc_ids], dtype=np.float32)

    # 1. 如果文档总数过于稀少 (< min_cluster_size)，直接将所有文档标记为 Outliers
    if len(doc_ids) < min_cluster_size:
        return [], doc_ids

    # 2. UMAP 降维：1024D → umap_n_components D
    # 关键：在高维空间 HDBSCAN 无法找到有意义的密度差异，UMAP 先将语义结构保留地映射到低维空间
    if _UMAP_AVAILABLE and len(doc_ids) > umap_n_components:
        reducer = umap_module.UMAP(
            n_components=umap_n_components,
            metric="cosine",          # 使用余弦距离，和 BGE-M3 向量的语义距离匹配
            n_neighbors=umap_n_neighbors,
            min_dist=0.0,             # 允许紧密聚合，有利于 HDBSCAN 发现密集簇
            random_state=42,          # 固定随机种子，保证每次运行结果一致
        )
        matrix_for_clustering = reducer.fit_transform(matrix_1024d)
    else:
        # 降级：UMAP 不可用时直接用原始向量（聚类质量会下降）
        if not _UMAP_AVAILABLE:
            print("[Warning] umap-learn 未安装，直接在 1024D 高维空间聚类（效果较差）。建议安装: pip install umap-learn")
        matrix_for_clustering = matrix_1024d

    # 3. HDBSCAN 在低维空间聚类
    # cluster_selection_method='leaf'：选取叶子节点簇，避免大簇吞噬小簇
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="leaf",
    )
    # labels: shape (N,) 的 int 数组，-1 代表 Outlier
    labels = clusterer.fit_predict(matrix_for_clustering)

    # 4. 按聚类 label 进行分组
    # zip(doc_ids, labels) 保证 doc_id 与 label 严格按下标对齐
    cluster_groups: Dict[int, List[int]] = {}
    outliers: List[int] = []

    for did, label in zip(doc_ids, labels):
        if label == -1:
            outliers.append(did)
        else:
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(did)

    # 5. 对每个形成的聚类簇，计算「原始 1024D 质心向量」并挑选代表文档
    # 注意：质心在原始 1024D 空间计算，而非 UMAP 降维后的空间，
    # 这样质心可以直接用于后续的余弦相似度点积运算（tag matching 和 confidence 计算）
    cluster_results: List[ClusterResult] = []

    for cid, c_dids in cluster_groups.items():
        c_vecs = np.array([doc_vectors[did] for did in c_dids], dtype=np.float32)

        # 5.1 计算 1024D 质心并 L2 归一化
        centroid = np.mean(c_vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # 5.2 计算簇内各文档与质心的余弦相似度，选出最近的 5 份代表性文档
        sims = np.clip(np.dot(c_vecs, centroid), 0.0, 1.0)
        sorted_indices = np.argsort(-sims)
        sorted_dids = [c_dids[idx] for idx in sorted_indices]
        rep_dids = sorted_dids[:min(5, len(sorted_dids))]

        cluster_results.append(
            ClusterResult(
                cluster_id=cid,
                doc_ids=c_dids,
                centroid_vector=centroid.astype(np.float32),
                representative_doc_ids=rep_dids,
            )
        )

    return cluster_results, outliers
