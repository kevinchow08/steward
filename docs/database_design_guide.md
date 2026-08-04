# 后端与数据库设计核心思维与实战指南

本指南梳理了在 `steward` 项目研发中沉淀的数据库表设计思维、主从关系推导法则、物理/逻辑外键、复合主键应用以及跨表查询逻辑，便于后续随时复习与查阅。

---

## 1. 从零推导表关系的“双向提问法” (Dual Questioning Method)

在动手写任何建表 SQL 之前，纯凭业务逻辑，对任意两个实体（如【实体 A】与【实体 B】）做双向提问：

### 提问法则：
1. **顺着问**：一份【实体 A】，可以对应多少个【实体 B】？
2. **反着问**：一份【实体 B】，可以对应多少个【实体 A】？

### 判定与落地规则：

| 双向提问结果 | 对应关系类型 | 主从关系判断 | 数据库物理落地动作 |
| :--- | :--- | :--- | :--- |
| **A : B = 1 : N** (或 N : 1) | **一对多 (1-to-N)** | **“多”的那一方（B）是从表** | 把“一”的 ID (`a_id`) 作为**外键**存入“多”的 B 表中。 |
| **A : B = N : M** | **多对多 (N-to-M)** | **A 和 B 都是主表** | 必须建立**第 3 张中间表 (Junction Table)**，同时包含 `a_id` 和 `b_id` 两个外键。 |
| **A : B = 1 : 1** | **一对一 (1-to-1)** | **任意指定一方为从表** | 把主表 ID 作为外键存入从表，并在外键列加 `UNIQUE` 唯一约束。 |

---

## 2. 字段约束设计的 4 项黄金 CheckList

针对新表中的每一个字段，依次过一遍这 4 项检查：

```text
建表约束思考清单 (Checklist)：

1. 【主键约束 PRIMARY KEY】:
   - 这个字段能作为这行数据的唯一身份 ID 吗？
   - 基础表用自增 id；中间关联表优先使用 (a_id, b_id) 复合主键。

2. 【非空约束 NOT NULL】:
   - 业务上这个字段允许“留空/不填”吗？
   - 关键凭证、名称、关联 ID 必须加 NOT NULL。

3. 【唯一约束 UNIQUE】:
   - 整个系统里允许出现两个一模一样的值吗？
   - 分类名称、标签名称、账号邮箱必须加 UNIQUE。

4. 【外键约束 FOREIGN KEY】:
   - 这个字段是不是引用了另一张主表的主键？
   - 只要存了 parent_id，必须加 FOREIGN KEY ... ON DELETE CASCADE 实现自动级联清理。
```

---

## 3. 逻辑外键 vs 物理外键

* **逻辑外键**：业务语义上的关联列（如 `document_tags` 表里的 `tag_id`）。任何跨表查询和业务拼装都依赖逻辑外键。
* **物理外键**：在 DDL 中声明 `FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE`。
* **核心结论**：**物理外键是用来服务逻辑外键的**。物理外键强行让数据库在底层替我们把关数据合法性，并在删除父表时自动清理子表垃圾数据（级联删除），保持系统高度可靠。

---

## 4. 复合主键 (Composite Primary Key) 的妙用

在中间表 `document_tags` 中：

```sql
CREATE TABLE document_tags (
    document_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    confidence REAL NOT NULL,
    PRIMARY KEY (document_id, tag_id),  -- 复合主键
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);
```

### 作用机制：
* `PRIMARY KEY (document_id, tag_id)` 将 **`document_id + tag_id` 的二元组合** 指定为主键。
* 单个 `document_id` 可以出现多次，单个 `tag_id` 也可以出现多次。
* **防止脏数据**：物理层面阻断完全相同的 `(1, 101)` 组合插入两次，避免一个文档被重复打上两个一模一样的标签。

---

## 5. 跨表查询与字典表拼装流程

字典表（如 `tags`、`categories`）只负责保存去重后的唯一名称。

### 从 SQLite 到最终 JSON 的拼装三步曲：

```text
1. 执行 SQL 跨表查询 (JOIN)：
   SELECT t.name 
   FROM document_tags dt 
   JOIN tags t ON t.id = dt.tag_id 
   WHERE dt.document_id = 1;

2. Python 接收从表查询结果并聚合：
   tag_list = ["周报", "Agent", "2026"]

3. 拼装为标准 JSON 契约：
   {
     "document_id": 1,
     "category": "工作文档",
     "tags": ["周报", "Agent", "2026"],
     "confidence": 0.92
   }
```

---

## 6. 高级架构模式：1:1 扩展表与物理/业务解耦设计

在 `steward` 项目中，`document_classifications` 表与 `documents` 表形成了标准的 **1:1 业务解耦扩展表** 模式：

```sql
CREATE TABLE document_classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL UNIQUE, -- UNIQUE 强制实现 1:1 对应
    category_id INTEGER NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    reasoning TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE
);
```

### 设计意图与解耦收益：

1. **物理扫描与 AI 语义解耦**：
   * `documents` 表属于 **物理硬件层**（只存文件路径 `path`、大小 `size`、修改时间 `mtime`）。只要文件在 Mac 硬盘上，这些数据保持客观不变。
   * `document_classifications` 表属于 **AI 语义推断层**（存储 `confidence` 置信度、`reasoning` 推理过程、打标时间）。
   * 今后即使升级端侧 AI 模型或修改分类 Prompt 重新打标，**只需重写分类扩展表，完全无需修改或污染基础物理文件表**。
2. **`UNIQUE` 约束保障业务规则**：
   * 通过 `document_id INTEGER NOT NULL UNIQUE` 唯一约束，在物理层面上保持了“一个文件只能拥有一个主分类”的业务铁律。

---

## 7. 多对多列表更新的“鬼魂脏数据”陷阱与“先删后加”策略

在更新 1 对 1 关系与多对多 (N:M) 标签列表时，物理更新策略有本质区别：

### 1. 单值覆盖 (1 对 1 主分类)
在 `document_classifications` 中，因为 `document_id` 带有 `UNIQUE` 约束，直接使用 `ON CONFLICT(document_id) DO UPDATE SET` 即可在原位置完成无痕覆盖，无需提前删除。

### 2. 列表替换 (N 对 M 标签列表) —— “鬼魂数据”陷阱
假定文档原先拥有标签 `['A', 'B', 'C']`，二次分类后变为了 `['B', 'D']`（移除了 `A` 和 `C`）：
* **错误的更新方式（只做 ON CONFLICT UPDATE）**：`B` 被更新，`D` 被插入，但旧的 `A` 和 `C` 依然会被留在数据库中，导致文档最终变成 `['A', 'B', 'C', 'D']` 幽灵脏数据！
* **正确的工程更新方式（先删后加原子事务）**：
  ```python
  with self.connection: # 开启原子事务
      # 1. 显式清空该文档的所有旧关联标签
      self.connection.execute("DELETE FROM document_tags WHERE document_id = ?", (doc_id,))
      
      # 2. 重新批量插入最新算出的标签列表
      for tag_name in new_tags:
          ...
  ```

> **事务防崩溃保证**：步骤 1 与 步骤 2 必须包裹在同一个数据库事务 (`with self.connection:`) 中。若在插入新标签过程中发生任何异常崩溃，前面的删除动作会自动回滚，绝不会造成标签数据丢失。

