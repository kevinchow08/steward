"""document 语义索引的 SQLite 持久化层。

这一层保存文件元数据、提取文本、文本片段和 embedding 向量。
它不负责调用模型，也不负责判断哪些文件最相关。
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "steward.db"


def _utc_now():
    """生成便于写入 SQLite 的 UTC 时间字符串。"""

    return datetime.now(timezone.utc).isoformat()


class DocumentIndex:
    """管理本地 SQLite 索引文件。"""

    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _create_tables(self):
        """第一次打开数据库时创建所需表；已有表不会被覆盖。"""

        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                model_name TEXT,
                model_dimension INTEGER,
                stats_json TEXT
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                size_bytes INTEGER,
                mtime_ns INTEGER,
                basic_type TEXT,
                is_present INTEGER NOT NULL DEFAULT 1,
                last_seen_run_id INTEGER,
                cluster_id INTEGER,
                FOREIGN KEY (last_seen_run_id) REFERENCES index_runs(id),
                FOREIGN KEY (cluster_id) REFERENCES semantic_clusters(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS extractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL UNIQUE,
                extractor_name TEXT,
                status TEXT NOT NULL,
                error TEXT,
                full_text TEXT NOT NULL DEFAULT '',
                char_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                extraction_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                UNIQUE (extraction_id, chunk_index),
                FOREIGN KEY (extraction_id) REFERENCES extractions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS embedding_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                normalized INTEGER NOT NULL,
                UNIQUE (model_name, dimension, normalized)
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER NOT NULL,
                model_id INTEGER NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (chunk_id, model_id),
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
                FOREIGN KEY (model_id) REFERENCES embedding_models(id) ON DELETE CASCADE
            );

            /* Week 3 新增：分类与打标签引擎核心表结构 */
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL UNIQUE,
                category_id INTEGER NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                reasoning TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS document_tags (
                document_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                confidence REAL NOT NULL,
                PRIMARY KEY (document_id, tag_id),
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            );

            /* Phase 3 算法簇解耦表：保存 HDBSCAN 发现的算法级聚类结构与 Tag Pool */
            CREATE TABLE IF NOT EXISTS semantic_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id INTEGER,
                centroid_vector BLOB NOT NULL,
                tag_pool_json TEXT NOT NULL,
                summary TEXT,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL
            );
            """
        )

        # 迁移：为 documents 表增加 cluster_id 字段 (外键关联 semantic_clusters)
        try:
            self.connection.execute("ALTER TABLE documents ADD COLUMN cluster_id INTEGER REFERENCES semantic_clusters(id)")
        except sqlite3.OperationalError:
            pass  # 如果 cluster_id 字段已存在，忽略错误

        self.connection.commit()

    def start_run(self, model_info=None):
        """记录一次索引任务的开始，返回 run id。"""

        model_name = model_info.model_name if model_info else None
        dimension = model_info.dimension if model_info else None

        cursor = self.connection.execute(
            """
            INSERT INTO index_runs
                (started_at, model_name, model_dimension)
            VALUES (?, ?, ?)
            """,
            (_utc_now(), model_name, dimension),
        )
        self.connection.commit()
        return cursor.lastrowid

    def finish_run(self, run_id, stats):
        """记录索引任务结束时间和统计数据。"""

        self.connection.execute(
            """
            UPDATE index_runs
            SET ended_at = ?, stats_json = ?
            WHERE id = ?
            """,
            (_utc_now(), json.dumps(stats, ensure_ascii=False), run_id),
        )
        self.connection.commit()

    def upsert_document(self, path, basic_type, run_id=None):
        """新增或更新文件元数据，返回 document id。"""

        file_path = Path(path).expanduser().resolve()
        stat = file_path.stat()

        self.connection.execute(
            """
            INSERT INTO documents
                (path, size_bytes, mtime_ns, basic_type, is_present, last_seen_run_id)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(path) DO UPDATE SET
                size_bytes = excluded.size_bytes,
                mtime_ns = excluded.mtime_ns,
                basic_type = excluded.basic_type,
                is_present = 1,
                last_seen_run_id = excluded.last_seen_run_id
            """,
            (str(file_path), stat.st_size, stat.st_mtime_ns, basic_type, run_id),
        )

        row = self.connection.execute(
            "SELECT id FROM documents WHERE path = ?",
            (str(file_path),),
        ).fetchone()
        self.connection.commit()
        return row["id"]

    def save_document_content(self, document_id, extraction, chunks, vectors, model_info):
        """保存一个文件的提取结果、chunk 和对应向量。

        同一个文件重新索引时，旧的提取结果、chunk 和向量会被级联删除，
        然后写入新版本，避免旧内容残留在搜索结果中。
        """

        if len(chunks) != len(vectors):
            raise ValueError("chunk 数量和向量数量不一致")

        model_id = self._get_or_create_model(model_info)

        with self.connection:
            self.connection.execute(
                "DELETE FROM extractions WHERE document_id = ?",
                (document_id,),
            )

            extraction_cursor = self.connection.execute(
                """
                INSERT INTO extractions
                    (document_id, extractor_name, status, error, full_text,
                     char_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    extraction.extractor,
                    extraction.status,
                    extraction.error,
                    extraction.text,
                    extraction.char_count,
                    _utc_now(),
                ),
            )
            extraction_id = extraction_cursor.lastrowid

            for chunk, vector in zip(chunks, vectors):
                chunk_cursor = self.connection.execute(
                    """
                    INSERT INTO chunks
                        (extraction_id, chunk_index, text, start_offset, end_offset)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        extraction_id,
                        chunk.index,
                        chunk.text,
                        chunk.start_offset,
                        chunk.end_offset,
                    ),
                )
                chunk_id = chunk_cursor.lastrowid
                vector_array = np.asarray(vector, dtype=np.float32)

                if vector_array.ndim != 1 or vector_array.shape[0] != model_info.dimension:
                    raise ValueError("向量维度与模型信息不一致")

                self.connection.execute(
                    """
                    INSERT INTO embeddings
                        (chunk_id, model_id, dimension, vector)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        model_id,
                        model_info.dimension,
                        vector_array.tobytes(),
                    ),
                )

    def get_model_id(self, model_info):
        """按模型信息查找已入库的模型记录。"""

        row = self.connection.execute(
            """
            SELECT id FROM embedding_models
            WHERE model_name = ? AND dimension = ? AND normalized = ?
            """,
            (
                model_info.model_name,
                model_info.dimension,
                int(model_info.normalized),
            ),
        ).fetchone()
        if row is None:
            return None
        return row["id"]

    def iter_search_vectors(self, model_id):
        """读取指定模型生成的全部 chunk 向量和展示所需元数据。"""

        cursor = self.connection.execute(
            """
            SELECT
                d.id AS document_id,
                d.path AS path,
                c.id AS chunk_id,
                c.chunk_index AS chunk_index,
                c.text AS chunk_text,
                e.dimension AS dimension,
                e.vector AS vector
            FROM embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN extractions x ON x.id = c.extraction_id
            JOIN documents d ON d.id = x.document_id
            WHERE e.model_id = ?
              AND d.is_present = 1
              AND x.status = 'success'
            """,
            (model_id,),
        )
        yield from cursor

    def _get_or_create_model(self, model_info):
        """取得模型记录；同一模型配置只保存一条。"""

        normalized = int(model_info.normalized)
        self.connection.execute(
            """
            INSERT OR IGNORE INTO embedding_models
                (model_name, dimension, normalized)
            VALUES (?, ?, ?)
            """,
            (model_info.model_name, model_info.dimension, normalized),
        )
        row = self.connection.execute(
            """
            SELECT id FROM embedding_models
            WHERE model_name = ? AND dimension = ? AND normalized = ?
            """,
            (model_info.model_name, model_info.dimension, normalized),
        ).fetchone()
        return row["id"]

    def clear_classifications(self):
        """清空旧有的分类、标签与算法簇历史数据，保证重新分类时无残存数据。"""
        with self.connection:
            # 必须最先解除 documents 对 semantic_clusters 的外键引用，否则 DELETE 会触发 FOREIGN KEY 约束报错！
            self.connection.execute("UPDATE documents SET cluster_id = NULL;")
            self.connection.execute("DELETE FROM document_tags;")
            self.connection.execute("DELETE FROM document_classifications;")
            self.connection.execute("DELETE FROM semantic_clusters;")
            self.connection.execute("DELETE FROM tags;")
            self.connection.execute("DELETE FROM categories;")

    def save_classification(
        self,
        document_id,
        category_name,
        tags=(),
        confidence=0.9,
        status="classified",
        reasoning="",
        cluster_id=None,
    ):
        """保存或更新文档的主分类与标签关联，并同步更新 documents.cluster_id 外键。"""

        category_id = self._get_or_create_category(category_name)
        now_str = _utc_now()

        # 防坑点 1：使用 with self.connection 开启 SQLite 原子事务！
        # 在 __enter__ 时自动开启事务，如果在后续多步写入/清理中发生异常崩溃，
        # __exit__ 会自动触发 rollback() 回滚所有删除与修改，防止产生数据半删半留的悬空状态；
        # 若正常执行结束，会自动触发 commit() 提交修改。
        with self.connection:
            if cluster_id is not None:
                self.connection.execute(
                    "UPDATE documents SET cluster_id = ? WHERE id = ?",
                    (cluster_id, document_id),
                )

            # 防坑点 2：使用 ON CONFLICT DO UPDATE (Upsert) 而非 INSERT OR REPLACE。
            # INSERT OR REPLACE 的底层物理动作是先物理删除旧行再插入新行（会导致自增 ID 发生变动/抖动，破坏外键引用）。
            # ON CONFLICT DO UPDATE 可以在保留原始主键 id 不变的前提下，在原位置做原地覆盖更新。
            self.connection.execute(
                """
                INSERT INTO document_classifications
                    (document_id, category_id, confidence, status, reasoning, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    category_id = excluded.category_id,
                    confidence = excluded.confidence,
                    status = excluded.status,
                    reasoning = excluded.reasoning,
                    created_at = excluded.created_at
                """,
                (document_id, category_id, confidence, status, reasoning, now_str),
            )

            # 防坑点 3：多对多标签列表的“先删后加”策略。
            # N:M 关联若直接 UPDATE，无法移除旧有但本次未传入的废弃标签（产生鬼魂数据）。
            # 必须先显式清空该文档的所有旧关联标签，再重新批量绑定最新算出的标签。
            self.connection.execute(
                "DELETE FROM document_tags WHERE document_id = ?",
                (document_id,),
            )

            # 批量绑定新的标签关联
            for item in tags:
                if isinstance(item, tuple):
                    tag_name, tag_conf = item[0], float(item[1])
                else:
                    tag_name, tag_conf = str(item), float(confidence)

                tag_name = tag_name.strip()
                if not tag_name:
                    continue
                tag_id = self._get_or_create_tag(tag_name)
                self.connection.execute(
                    """
                    INSERT INTO document_tags (document_id, tag_id, confidence)
                    VALUES (?, ?, ?)
                    ON CONFLICT(document_id, tag_id) DO UPDATE SET
                        confidence = excluded.confidence
                    """,
                    (document_id, tag_id, tag_conf),
                )

    def iter_classifications(self):
        """读取所有文档的主分类、状态、置信度及标签列表。"""

        cursor = self.connection.execute(
            """
            SELECT
                d.id AS document_id,
                d.path AS path,
                d.basic_type AS basic_type,
                c.name AS category_name,
                dc.confidence AS confidence,
                dc.status AS status,
                dc.reasoning AS reasoning,
                dc.created_at AS created_at
            FROM documents d
            JOIN document_classifications dc ON dc.document_id = d.id
            JOIN categories c ON c.id = dc.category_id
            WHERE d.is_present = 1
            ORDER BY dc.created_at DESC
            """
        )

        for row in cursor.fetchall():
            doc_id = row["document_id"]
            # 查出该文档绑定的所有标签名称
            tag_rows = self.connection.execute(
                """
                SELECT t.name
                FROM document_tags dt
                JOIN tags t ON t.id = dt.tag_id
                WHERE dt.document_id = ?
                ORDER BY t.name ASC
                """,
                (doc_id,),
            ).fetchall()

            tags_list = [tr["name"] for tr in tag_rows]

            yield {
                "document_id": doc_id,
                "path": row["path"],
                "basic_type": row["basic_type"],
                "category": row["category_name"],
                "confidence": row["confidence"],
                "status": row["status"],
                "reasoning": row["reasoning"],
                "tags": tags_list,
                "created_at": row["created_at"],
            }

    def _get_or_create_category(self, category_name):
        """获取或创建主分类记录，返回 category_id。"""

        category_name = category_name.strip() or "unclassified"
        now_str = _utc_now()
        self.connection.execute(
            """
            INSERT OR IGNORE INTO categories (name, created_at)
            VALUES (?, ?)
            """,
            (category_name, now_str),
        )
        row = self.connection.execute(
            "SELECT id FROM categories WHERE name = ?",
            (category_name,),
        ).fetchone()
        return row["id"]

    def _get_or_create_tag(self, tag_name):
        """获取或创建标签字典记录，返回 tag_id。"""

        tag_name = tag_name.strip()
        now_str = _utc_now()
        self.connection.execute(
            """
            INSERT OR IGNORE INTO tags (name, created_at)
            VALUES (?, ?)
            """,
            (tag_name, now_str),
        )
        row = self.connection.execute(
            "SELECT id FROM tags WHERE name = ?",
            (tag_name,),
        ).fetchone()
        return row["id"]

    def save_semantic_cluster(
        self,
        category_id: int,
        centroid_vector: np.ndarray,
        tag_pool: list[str],
        summary: str = "",
        confidence: float = 0.90,
    ) -> int:
        """保存 HDBSCAN 算法层产生的解耦语义簇 (semantic_cluster)，返回 cluster_id。"""
        now_str = _utc_now()
        vector_blob = centroid_vector.astype(np.float32).tobytes()
        tag_pool_json = json.dumps(tag_pool, ensure_ascii=False)

        cursor = self.connection.execute(
            """
            INSERT INTO semantic_clusters
                (category_id, centroid_vector, tag_pool_json, summary, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (category_id, vector_blob, tag_pool_json, summary, confidence, now_str),
        )
        return cursor.lastrowid

