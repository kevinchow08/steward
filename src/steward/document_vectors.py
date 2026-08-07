"""文档级向量聚合模块 (Document Vector Weighted Pooling)

将同一文档下的多个 1024 维 Chunk 切片向量，通过字符长度加权平均（Weighted Pooling）
与 L2 归一化，合成为代表该文档全局语义的唯一 1024 维向量。
纯依赖 NumPy 矩阵运算，零模型开销，毫秒级完成。
"""

import sqlite3
from typing import Dict, List, Tuple
import numpy as np

from steward.document_index import DEFAULT_DB_PATH, DocumentIndex


def compute_document_vector(
    chunk_vectors: List[np.ndarray],
    chunk_lengths: List[int],
) -> np.ndarray:
    """对单个文档的多个切片向量进行加权池化与 L2 归一化。

    :param chunk_vectors: 切片向量列表，每个向量形状为 (1024,)
    :param chunk_lengths: 每个切片的字符长度列表
    :return: 代表该文档的 1024 维归一化向量 (1024,)
    """

    if not chunk_vectors:
        raise ValueError("切片向量列表不能为空")

    vectors = np.array(chunk_vectors, dtype=np.float32)  # 形状: (N, 1024)
    weights = np.array(chunk_lengths, dtype=np.float32)  # 形状: (N,)

    # 1. 权重归一化：若长度总和为 0，则采用均匀平均
    total_weight = np.sum(weights)
    if total_weight > 0:
        normalized_weights = weights / total_weight
    else:
        normalized_weights = np.ones(len(weights), dtype=np.float32) / len(weights)

    # 2. 加权平均池化: (N, 1024) 点积 (N, 1) -> (1024,)
    doc_vec = np.average(vectors, weights=normalized_weights, axis=0)

    # 3. L2 范数归一化 (保证 ||v|| = 1.0，便于后续余弦点积计算)
    norm = np.linalg.norm(doc_vec)
    if norm > 0:
        doc_vec = doc_vec / norm

    return doc_vec.astype(np.float32)


def get_all_document_vectors(db_path=DEFAULT_DB_PATH) -> Dict[int, np.ndarray]:
    """从 SQLite 中加载已索引文档的所有 Chunk 向量并批量合成为文档向量。

    :return: 字典 {document_id: document_vector}
    """

    with DocumentIndex(db_path) as index:
        # 一次性关联查询：取 document_id, chunk_id, 文本长度, 向量 BLOB
        cursor = index.connection.execute(
            """
            SELECT
                x.document_id AS doc_id,
                c.id AS chunk_id,
                LENGTH(c.text) AS char_len,
                e.vector AS vector_blob
            FROM chunks c
            JOIN extractions x ON x.id = c.extraction_id
            JOIN embeddings e ON e.chunk_id = c.id
            JOIN documents d ON d.id = x.document_id
            WHERE d.is_present = 1
              AND x.status = 'success'
            ORDER BY x.document_id, c.chunk_index
            """
        )
        rows = cursor.fetchall()

    # 按 document_id 分组收集
    doc_chunks: Dict[int, List[Tuple[np.ndarray, int]]] = {}
    for r in rows:
        doc_id = r["doc_id"]
        char_len = r["char_len"]
        vec = np.frombuffer(r["vector_blob"], dtype=np.float32)
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
        doc_chunks[doc_id].append((vec, char_len))

    # 批量计算每个文档的 Weighted Pooling 向量
    doc_vectors: Dict[int, np.ndarray] = {}
    for doc_id, chunk_list in doc_chunks.items():
        vecs = [item[0] for item in chunk_list]
        lens = [item[1] for item in chunk_list]
        doc_vectors[doc_id] = compute_document_vector(vecs, lens)

    return doc_vectors
