"""无监督语义聚类模块 (HDBSCAN Unsupervised Semantic Clustering)

利用基于密度的层次聚类算法 (HDBSCAN)，从 1024 维文档向量空间中自动发现文件的自然高密度主题簇。
无需硬编码预设簇数量 K，自动将低密度离群文件判定为 Noise/Outliers (label = -1)。
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import HDBSCAN


@dataclass
class ClusterResult:
    """单个聚类簇的数据结构。"""

    cluster_id: int
    doc_ids: List[int]
    centroid_vector: np.ndarray  # 该簇的中心归一化向量 (1024,)
    representative_doc_ids: List[int]  # 距离质心最近的 3~5 份代表性文档 ID


def cluster_document_vectors(
    doc_vectors: Dict[int, np.ndarray],
    min_cluster_size: int = 3,
    min_samples: int = 2,
    metric: str = "euclidean",
) -> Tuple[List[ClusterResult], List[int]]:
    """对全量文档向量进行 HDBSCAN 无监督自适应聚类。

    :param doc_vectors: 字典 {doc_id: document_vector (1024,)}
    :param min_cluster_size: 一个自适应语义簇所需的最小文档数量 (默认 3 份)
    :param min_samples: 确定核心点邻域的最小样本数
    :param metric: 距离度量标准 (由于向量已 L2 归一化，欧氏距离与余弦距离单调等价)
    :return: (已形成的聚类簇列表, 离群噪点文档 ID 列表 outliers)
    """

    if not doc_vectors:
        return [], []

    doc_ids = list(doc_vectors.keys())
    # 组合为 2D 矩阵 (N, 1024)
    matrix_2d = np.array([doc_vectors[did] for did in doc_ids], dtype=np.float32)

    # 1. 如果文档总数过于稀少 (< min_cluster_size)，直接将所有文档标记为 Outliers
    if len(doc_ids) < min_cluster_size:
        return [], doc_ids

    # 2. 运行 scikit-learn 原生 HDBSCAN 算法
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    # labels 是一个 Shape 为 (N,) 的 1D int64 NumPy 数组，例如 array([0, 0, 1, 2, -1, 0, ...])
    # 数组第 i 个位置的值即为 matrix_2d 第 i 行文档所属的簇 ID (-1 代表噪声 Outlier)
    labels = clusterer.fit_predict(matrix_2d)

    # 3. 按聚类 label 进行分组
    # 注意：因为 matrix_2d 是严格按照 doc_ids 的下标顺序 0..N-1 建立的，
    # 所以 matrix_2d 的第 i 行就是 doc_ids[i]，其聚类结果就是 labels[i]。
    # 使用 zip(doc_ids, labels) 可以 100% 物理保证 did 与 label 的位置严格对齐！
    cluster_groups: Dict[int, List[int]] = {}
    outliers: List[int] = []

    for did, label in zip(doc_ids, labels):
        if label == -1:
            outliers.append(did)
        else:
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(did)

    # 4. 对每个形成的聚类簇，计算质心向量并挑选代表文档 (Representative Documents)
    cluster_results: List[ClusterResult] = []

    for cid, c_dids in cluster_groups.items():
        c_vecs = np.array([doc_vectors[did] for did in c_dids], dtype=np.float32)

        # 4.1 计算质心向量并做 L2 归一化
        centroid = np.mean(c_vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # 4.2 计算簇内各个文档向量与质心的余弦相似度，并做 [0.0, 1.0] 数值截断防溢出
        sims = np.clip(np.dot(c_vecs, centroid), 0.0, 1.0)
        sorted_indices = np.argsort(-sims)  # 降序索引
        sorted_dids = [c_dids[idx] for idx in sorted_indices]

        # 挑选距离质心最近的前 5 份文档作为代表性文档 (供后续 LLM 理解采样)
        rep_dids = sorted_dids[: min(5, len(sorted_dids))]

        cluster_results.append(
            ClusterResult(
                cluster_id=cid,
                doc_ids=c_dids,
                centroid_vector=centroid.astype(np.float32),
                representative_doc_ids=rep_dids,
            )
        )

    return cluster_results, outliers
