"""文档级向量聚合模块 (Document Vector Weighted Pooling)

将同一文档下的多个 1024 维 Chunk 切片向量，通过字符长度加权平均（Weighted Pooling）
与 L2 归一化，合成为代表该文档全局语义的唯一 1024 维向量。
纯依赖 NumPy 矩阵运算，零模型开销，毫秒级完成。
"""

from pathlib import Path
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


def get_all_document_vectors(index=None, db_path: Path = DEFAULT_DB_PATH, model_id: int = 1) -> Dict[int, np.ndarray]:
    """从数据库检索所有成功提取文本的文档及其切片向量，并通过 Pooling 算出一一对应的文档向量。

    :param index: DocumentIndex 实例 (若外部传入则直接复用其连接，生命周期由外部管理)
    :param db_path: SQLite 数据库文件路径 (当 index 为 None 时自动创建临时连接)
    :param model_id: Embedding 模型 ID (默认 1，代表 bge-m3)
    :return: 字典 {document_id: (1024,) float32 np.ndarray}
    """
    if index:
        return _query_and_compute_vectors(index.connection, model_id)

    with DocumentIndex(db_path) as temporary_index:
        return _query_and_compute_vectors(temporary_index.connection, model_id)


def _query_and_compute_vectors(connection, model_id: int) -> Dict[int, np.ndarray]:
    """内部 Query 执行与加权 Pooling。"""
    query = """
        SELECT
            x.document_id AS doc_id,
            c.id AS chunk_id,
            LENGTH(c.text) AS char_len,
            e.vector AS vector_blob
        FROM chunks c
        JOIN extractions x ON x.id = c.extraction_id
        JOIN embeddings e ON e.chunk_id = c.id
        JOIN documents d ON d.id = x.document_id
        WHERE e.model_id = ? AND x.status = 'success'
        ORDER BY x.document_id, c.chunk_index
    """

    cursor = connection.execute(query, (model_id,))

    doc_chunks = {}
    for row in cursor.fetchall():
        doc_id = row["doc_id"]
        char_len = row["char_len"]
        vec = np.frombuffer(row["vector_blob"], dtype=np.float32)

        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
        doc_chunks[doc_id].append((vec, char_len))

    doc_vectors = {}
    for doc_id, chunks in doc_chunks.items():
        chunk_vecs = [c[0] for c in chunks]
        chunk_lens = [c[1] for c in chunks]
        doc_vectors[doc_id] = compute_document_vector(chunk_vecs, chunk_lens)

    return doc_vectors
