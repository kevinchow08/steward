"""document 语义索引的 SQLite 持久化层。

这一层保存文件元数据、提取文本、文本片段和 embedding 向量。
它不负责调用模型，也不负责判断哪些文件最相关。
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "output" / "steward.db"


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
                FOREIGN KEY (last_seen_run_id) REFERENCES index_runs(id)
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
            """
        )
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
