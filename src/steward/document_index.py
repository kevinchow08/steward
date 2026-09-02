"""document 语义索引的 SQLite 持久化层。

这一层保存文件元数据、提取文本、文本片段和 embedding 向量。
它不负责调用模型，也不负责判断哪些文件最相关。
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# 之前这里指向项目源码目录本身（.../steward/steward.db），是开发阶段图方便的写法，
# 真要打包分发给用户就不对了——要么落在安装包内部（可能没有写权限，或者升级/卸载时
# 被清空），不是一个适合"持续积累、跨目录/跨盘共享"的用户数据存储位置。改成 macOS
# 标准的用户级应用数据目录，这样不管这次 index 的是 ~/Documents 还是某个外置盘，
# 只要不手动传 --db，天然都会写进同一个共享数据库，不需要用户自己记着保持一致。
# 这是 macOS 专属路径约定，以后要上 Windows/Linux 得换成对应平台的标准位置。
DEFAULT_DB_PATH = Path.home() / "Library" / "Application Support" / "Steward" / "steward.db"


def _utc_now():
    """生成便于写入 SQLite 的 UTC 时间字符串。"""

    return datetime.now(timezone.utc).isoformat()


def _derive_volume_label(resolved_path):
    """从绝对路径推断这个文件躺在哪个盘上。macOS 外置卷统一挂载在
    /Volumes/<卷名>/... 下面；不是这个前缀的，就是内置主硬盘，没有一个专门的
    挂载点路径前缀，统一标成"本机"。这是 macOS 专属的路径约定，以后要上
    Windows/Linux 得换成对应平台的挂载点判断逻辑（比如 Windows 的盘符）。
    """
    parts = resolved_path.parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        return parts[2]
    return "本机"


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
                target_dir TEXT,
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
                volume_label TEXT,
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

            /* 混合检索用的关键词索引（FTS5 全文检索虚拟表），跟 chunks 表镜像，
               rowid 直接复用 chunks.id，查询时靠这个字段跟 chunks/documents 关联，
               不用额外维护一张映射表。tokenize='trigram'：按三字符一组滑窗匹配，
               不是真正的分词——中文没有空格分词，默认的 unicode61 分词器会把一整句
               话当成一个词，等于没法做关键词匹配；trigram 不需要理解语言，中英文
               都能覆盖，代价是查询词必须至少 3 个字符才能匹配上（真实测过，2 个字
               的查询词一个字符组都凑不出来，天然搜不到——这是设计上的已知代价，
               不是 bug，混合检索的另一路向量检索没有这个长度限制，能接住这类短查询）。
               FTS5 虚拟表不受外键 ON DELETE CASCADE 管辖，chunks 被级联删除时这张表
               不会跟着自动清空，需要在 save_document_content() 里手动同步维护，见那边
               的注释。 */
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='trigram');

            /* Week 3 新增：打标签引擎核心表结构。这张表最早叫 document_classifications，
               还配了一张 categories 表、一个 category_id 外键，维护一套分类体系，V3
               试了两周后被拿掉了——真实数据反复证明"每份文件收敛到唯一分类"这个目标
               本身跟文件天然多归属、边界模糊的性质冲突，具体过程见
               docs/dynamic_classification_architecture.md。现在这张表只存打标签结果
               （reasoning + 置信度 + 状态），不再关联分类，改名 tagging_results，不留
               "classification"这种名不副实的痕迹；也跟下面 document_tags（文档↔标签
               的多对多桥接表）区分开——那张表是"文档有哪些标签"，这张表是"这个文档打
               标签这件事本身的元数据"，一对一，不是同一件事，别看名字像就以为能合并。 */
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tagging_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL UNIQUE,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                reasoning TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS document_tags (
                document_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                confidence REAL NOT NULL,
                PRIMARY KEY (document_id, tag_id),
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            );
            """
        )

        self.connection.commit()

    def start_run(self, model_info=None, target_dir=None):
        """记录一次索引任务的开始，返回 run id。

        target_dir：这次运行扫描的根目录，记下来是为了让"幽灵条目清理"能限定
        范围——统一数据库下可能同时存有其它目录/盘的记录，这次运行只应该影响
        它自己扫描的这棵子树，不能全库扫。见 mark_missing_as_absent()。
        """

        model_name = model_info.model_name if model_info else None
        dimension = model_info.dimension if model_info else None

        cursor = self.connection.execute(
            """
            INSERT INTO index_runs
                (started_at, target_dir, model_name, model_dimension)
            VALUES (?, ?, ?, ?)
            """,
            (_utc_now(), target_dir, model_name, dimension),
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
        volume_label = _derive_volume_label(file_path)

        self.connection.execute(
            """
            INSERT INTO documents
                (path, size_bytes, mtime_ns, basic_type, volume_label, is_present, last_seen_run_id)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(path) DO UPDATE SET
                size_bytes = excluded.size_bytes,
                mtime_ns = excluded.mtime_ns,
                basic_type = excluded.basic_type,
                volume_label = excluded.volume_label,
                is_present = 1,
                last_seen_run_id = excluded.last_seen_run_id
            """,
            (str(file_path), stat.st_size, stat.st_mtime_ns, basic_type, volume_label, run_id),
        )

        row = self.connection.execute(
            "SELECT id FROM documents WHERE path = ?",
            (str(file_path),),
        ).fetchone()
        self.connection.commit()
        return row["id"]

    def get_up_to_date_documents(self, model_id):
        """返回 {path: (size_bytes, mtime_ns)}，只包含"在当前 embedding 模型下
        已经有完整向量"的文档——供增量索引判断"这个文件要不要跳过重新处理"。

        model_id 为 None（这个模型配置从来没在这个库里用过）时返回空字典，
        代表没有任何文件能被跳过，全部老老实实走一遍完整流程——这是安全的
        默认行为，不会因为"以为没变"而漏处理。

        注意：只覆盖 extractions.status='success' 的文件（真正切过片、生成过
        向量的）。status 是 no_text/error 的文件不会出现在这里，意味着它们
        每次都会被重新提取一遍——这些文件重新提取的成本远低于"切片+向量化"，
        不值得为它们单独再加一层判断逻辑。
        """
        if model_id is None:
            return {}
        cursor = self.connection.execute(
            """
            SELECT DISTINCT d.path AS path, d.size_bytes AS size_bytes, d.mtime_ns AS mtime_ns
            FROM documents d
            JOIN extractions x ON x.document_id = d.id
            JOIN chunks c ON c.extraction_id = x.id
            JOIN embeddings e ON e.chunk_id = c.id
            WHERE e.model_id = ? AND x.status = 'success'
            """,
            (model_id,),
        )
        return {row["path"]: (row["size_bytes"], row["mtime_ns"]) for row in cursor.fetchall()}

    def mark_missing_as_absent(self, target_dir, run_id):
        """把 target_dir 这棵子树下、这次运行没扫到的旧记录标记为不存在
        （is_present=0），返回被标记的行数。

        path = target_dir 本身：目录自己就是一条记录的情况（比如代码项目）。
        path LIKE target_dir/%：目录底下的所有文件。用路径前缀限定范围，
        不能全库扫——统一数据库下可能同时存着其它目录/盘的记录，这次运行
        不能波及它们。

        last_seen_run_id != run_id：upsert_document() 对每个真被扫到的文件
        都会把 last_seen_run_id 刷新成这次的 run_id，所以只要不等于这次的
        run_id，就代表这次扫描没碰到它，判定为消失。

        ESCAPE '\\' + escaped：真实文件夹名字常带 % 或 _（比如"周凯文_住房
        补贴相关"），这两个字符在 SQL LIKE 里是通配符，不转义会被误当成
        通配符匹配到不该匹配的路径。
        """
        prefix = target_dir.rstrip("/")
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE documents
                SET is_present = 0
                WHERE is_present = 1
                  AND (path = ? OR path LIKE ? ESCAPE '\\')
                  AND (last_seen_run_id IS NULL OR last_seen_run_id != ?)
                """,
                (prefix, escaped + "/%", run_id),
            )
            return cursor.rowcount

    def save_project_extraction(self, document_id, full_text):
        """保存一个"项目"单位（代码仓库根目录）的自描述内容（README/package.json 摘要拼接）。

        项目不像文档那样需要切片+embedding（数量少，不用参与向量检索），只需要一份
        "正文"落进 extractions 表，让它能走现有的 tag 报告展示逻辑，不用新建表。
        """
        with self.connection:
            self.connection.execute(
                "DELETE FROM extractions WHERE document_id = ?",
                (document_id,),
            )
            self.connection.execute(
                """
                INSERT INTO extractions
                    (document_id, extractor_name, status, error, full_text,
                     char_count, created_at)
                VALUES (?, ?, 'success', NULL, ?, ?, ?)
                """,
                (document_id, "project_readme", full_text, len(full_text), _utc_now()),
            )

    def save_document_content(self, document_id, extraction, chunks, vectors, model_info):
        """保存一个文件的提取结果、chunk 和对应向量。

        同一个文件重新索引时，旧的提取结果、chunk 和向量会被级联删除，
        然后写入新版本，避免旧内容残留在搜索结果中。
        """

        if len(chunks) != len(vectors):
            raise ValueError("chunk 数量和向量数量不一致")

        model_id = self._get_or_create_model(model_info)

        with self.connection:
            # chunks_fts 是 FTS5 虚拟表，不受外键 ON DELETE CASCADE 管辖——下面
            # "DELETE FROM extractions"会级联删掉旧的 chunks/embeddings，但不会
            # 动 chunks_fts，不手动清理的话，旧 chunk 的关键词索引会变成孤儿数据
            # 一直堆积。必须在级联删除之前先查出旧 chunk id，删掉之后就再也查不到了。
            old_chunk_ids = [
                row["id"] for row in self.connection.execute(
                    """
                    SELECT c.id AS id FROM chunks c
                    JOIN extractions x ON x.id = c.extraction_id
                    WHERE x.document_id = ?
                    """,
                    (document_id,),
                )
            ]
            if old_chunk_ids:
                self.connection.executemany(
                    "DELETE FROM chunks_fts WHERE rowid = ?",
                    [(cid,) for cid in old_chunk_ids],
                )

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

                # rowid 直接复用 chunk_id，查询时靠这个字段跟 chunks 表关联，
                # 不用另外维护一张"chunk_id -> fts_rowid"的映射表。
                self.connection.execute(
                    "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
                    (chunk_id, chunk.text),
                )

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

    def search_keyword_chunks(self, query):
        """用 FTS5 关键词检索，按 bm25 分数从好到坏排序，返回命中的 chunk 及
        所属文档信息。bm25 分数越小代表匹配度越高（这是 FTS5 的既定约定，
        不是"数值越大越好"的常见直觉，真实测过验证过，见 semantic_search.py
        混合检索那部分的注释）。

        不在这里限制返回条数——同一个文档可能有好几个 chunk 命中，过早在
        chunk 级别截断，可能会把"命中了很多个 chunk、按文档聚合起来其实排名
        很靠前"的文档提前挤掉。要不要截断、截多少，交给调用方在按文档聚合、
        算出最终排序之后再做。

        FTS5 的 MATCH 查询语法里 `"`、`*`、AND/OR/NOT 这些是有特殊含义的保留
        字符/关键词，用户的自然语言查询里凑巧出现是完全可能的（比如一句话里
        真的有个双引号），这种情况下 MATCH 会抛语法错误——捕获它、返回空列表，
        让关键词检索这一路安静地不参与这次排序，不能因为这个让整个混合检索
        崩掉（向量检索那一路完全不受这个语法限制，可以正常兜底）。
        """
        try:
            cursor = self.connection.execute(
                """
                SELECT c.id AS chunk_id, d.id AS document_id, d.path AS path,
                       c.chunk_index AS chunk_index, c.text AS chunk_text,
                       bm25(chunks_fts) AS bm25_score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                JOIN extractions x ON x.id = c.extraction_id
                JOIN documents d ON d.id = x.document_id
                WHERE chunks_fts MATCH ? AND d.is_present = 1 AND x.status = 'success'
                ORDER BY bm25_score ASC
                """,
                (query,),
            )
            return cursor.fetchall()
        except sqlite3.OperationalError:
            return []

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

    def clear_tagging_results(self):
        """清空旧有的打标签历史数据，保证重新打标签时无残存数据。"""
        with self.connection:
            self.connection.execute("DELETE FROM document_tags;")
            self.connection.execute("DELETE FROM tagging_results;")
            self.connection.execute("DELETE FROM tags;")

    def save_tagging_result(
        self,
        document_id,
        tags=(),
        confidence=0.9,
        status="tagged",
        reasoning="",
    ):
        """保存或更新文档的打标签结果（reasoning + 标签关联），不再关联分类。"""

        now_str = _utc_now()

        # 防坑点 1：使用 with self.connection 开启 SQLite 原子事务！
        # 保证的是"这一个文档的打标签结果 + 标签关联"这一组写入要么全部成功、
        # 要么全部不生效，不会出现"存了 tagging_results 但标签关联没写全"这种
        # 半吊子状态。注意这个保证只到"单个文档"这个粒度——之前这里的注释错误
        # 地暗示了"整个批量打标签过程"的原子性，实测验证过那个说法不成立：
        # sqlite3 的 `with conn:` 不支持真正的嵌套事务，如果调用方（比如批量
        # 循环）在外面又包了一层 `with index.connection:`，内层这个 with 退出时
        # 会提前把已经处理过的部分真正提交到磁盘，不会等到外层一起处理完才
        # commit/rollback。所以调用这个方法的地方不应该再在外面包一层
        # `with index.connection:` 期待"批量全部提交或全部回滚"，见
        # tagging.py 里 Step 5 持久化那段的注释。
        with self.connection:
            # 防坑点 2：使用 ON CONFLICT DO UPDATE (Upsert) 而非 INSERT OR REPLACE。
            # INSERT OR REPLACE 的底层物理动作是先物理删除旧行再插入新行（会导致自增 ID 发生变动/抖动，破坏外键引用）。
            # ON CONFLICT DO UPDATE 可以在保留原始主键 id 不变的前提下，在原位置做原地覆盖更新。
            self.connection.execute(
                """
                INSERT INTO tagging_results
                    (document_id, confidence, status, reasoning, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    status = excluded.status,
                    reasoning = excluded.reasoning,
                    created_at = excluded.created_at
                """,
                (document_id, confidence, status, reasoning, now_str),
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

    def iter_tagging_results(self):
        """读取所有文档的打标签状态、置信度、reasoning 及标签列表。"""

        cursor = self.connection.execute(
            """
            SELECT
                d.id AS document_id,
                d.path AS path,
                d.basic_type AS basic_type,
                dc.confidence AS confidence,
                dc.status AS status,
                dc.reasoning AS reasoning,
                dc.created_at AS created_at
            FROM documents d
            JOIN tagging_results dc ON dc.document_id = d.id
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
                "confidence": row["confidence"],
                "status": row["status"],
                "reasoning": row["reasoning"],
                "tags": tags_list,
                "created_at": row["created_at"],
            }

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


