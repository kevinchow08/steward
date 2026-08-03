# 端侧向量、NumPy 与数据库持久化核心速查指南

本指南梳理了 `steward` 项目中涉及的向量物理含义、NumPy 数据结构、数学相似度算法以及 SQLite 存储转换机制，便于后续开发中随时查阅。

---

## 1. 向量与维度 (Vector & Dimension)

- **向量 (Vector)**：在 AI / 语义检索中，向量是一个由浮点数组成的一维数组。模型通过把文本投影到高维几何空间，用数值来表示文本的抽象语义。
- **维度 (Dimension)**：指的是数组中**元素的总个数**。
  - 本项目使用 `BAAI/bge-m3` 模型，其生成的每个向量固定包含 **1024 个浮点数**（即 1024 维）。
  - 每一个维度代表文本在某个抽象语义方向上的分量。

---

## 2. NumPy 核心结构：`ndim` 与 `shape`

NumPy 的 `ndarray` 是 Python 中处理向量与矩阵的标准载体。

### 维度区分表

| 维度概念     | `ndim` (括号层数)  | `shape` 结构示例 | 现实对应物理含义                             |
| :----------- | :----------------- | :--------------- | :------------------------------------------- |
| **一维向量** | `1` (1层 `[]`)     | `(1024,)`        | **1 句文本/片段** 的向量（包含 1024 个元素） |
| **二维矩阵** | `2` (2层 `[[]]`)   | `(16, 1024)`     | **16 段文本** 的批量向量（16 行，1024 列）   |
| **三维张量** | `3` (3层 `[[[]]]`) | `(2, 16, 1024)`  | **2 个批次**，每个批次包含 16 段 1024 维向量 |

> **关键易错点**：
>
> - `shape` 的顺序遵循线性代数习惯：**`(行数 Rows, 列数 Columns)`**。
> - `(1024,)` 表示纯一维数组（只有元素个数，没有行/列概念）。
> - `(1, 1024)` 表示 1 行 1024 列的二维矩阵。
> - 代码中持久化单个向量时，校验 `vector_array.ndim == 1` 可以防止混入二维矩阵数据。

### NumPy 数组真实长相代码示例

#### 1. 单个文本片段的一维向量 (`shape = (1024,)`, `ndim = 1`)

只有 1 层中括号 `[]`：

```python
array([ 0.0123041, -0.4512399,  0.8812301, ..., -0.0091245], dtype=float32)
```

#### 2. 批量 16 段文本的二维矩阵 (`shape = (16, 1024)`, `ndim = 2`)

有 2 层中括号 `[[]]`，外层包含 16 行：

```python
array([
    [ 0.0123041, -0.4512399,  0.8812301, ..., -0.0091245],  # 第 1 段文本的向量
    [ 0.0561288,  0.1123945, -0.3412099, ...,  0.0128945],  # 第 2 段文本的向量
    ...
    [-0.1023941,  0.5541029,  0.1239412, ..., -0.4512049]   # 第 16 段文本的向量
], dtype=float32)
```

---

## 3. 模型 Encode 机制与 `batch_size` 设定

在 `embeddings.py` 的 `LocalEmbedder` 中，模型推理分为两类设定：

### `Batch` (批次) 的通俗物理含义

`batch_size` 并不是限制输入的文本总数，而是控制 **CPU/GPU 内存中单趟并行计算的容量上限**。

* **`texts` (待处理包裹)**：一篇长文档切分出来的**所有片段**（比如 `len(texts) = 50` 段）。
* **`batch_size = 16` (后备箱容量)**：规定模型一趟最多只能装载 16 段文本进行矩阵运算。

#### 50 段文本分批推理流图解：
```text
输入 texts (50 段文本)
  ├── 第 1 趟 (Batch 1): 处理 texts[0:16]  (16 段) ──> 算出 16 个向量
  ├── 第 2 趟 (Batch 2): 处理 texts[16:32] (16 段) ──> 算出 16 个向量
  ├── 第 3 趟 (Batch 3): 处理 texts[32:48] (16 段) ──> 算出 16 个向量
  └── 第 4 趟 (Batch 4): 处理 texts[48:50] ( 2 段) ──> 算出  2 个向量
                                                          │
                                                          ▼ 自动拼接整合
                                        返回 shape = (50, 1024) 的 NumPy 矩阵
```

* **防内存溢出 (OOM)**：避免在处理上千段文本时一次性灌入显存导致程序崩溃。
* **极佳吞吐量**：利用硬件多核矩阵并行处理，比单句逐一推理快 5~10 倍。

> **提示**：`bge-m3` 支持非对称搜索（Asymmetric Search）。编码文档时优先调用 `encode_document`，编码查询时优先调用 `encode_query`，以获得最佳语义匹配效果。

---

## 3. 模长、归一化与余弦相似度

### 1. 模长 (Magnitude / Norm)

向量在几何空间中的绝对箭头长度。计算公式为各分量平方和开根号：

$$\text{Length} = \sqrt{x_1^2 + x_2^2 + \dots + x_{1024}^2}$$

在 NumPy 中使用 `np.linalg.norm(vector)` 计算。

### 2. 归一化 (Normalization)

- **本质**：保持向量的方向角度完全不变，将其几何长度（模长）同比例缩放为 **`1.0`**。
- **作用**：剔除文章长短、字数多少对向量长度的干扰，**只纯粹比较文本的主题方向**。
- 在项目配置 `normalize_embeddings=True` 后，所有向量的模长均为 `1.0`。

### 3. 余弦相似度 (Cosine Similarity)

几何公式为两向量夹角 $\theta$ 的余弦值：

$$\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$

- **夹角 $\theta = 0^\circ$**（方向完全一致，高度相关）：$\cos(0^\circ) = 1.0$
- **夹角 $\theta = 90^\circ$**（方向互相垂直，毫无关联）：$\cos(90^\circ) = 0.0$

### 4. 归一化后的计算优化

当向量 $\vec{A}$ 和 $\vec{B}$ 均已归一化（模长 $\|\vec{A}\| = 1, \|\vec{B}\| = 1$）时，分母为 1：

$$\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{1 \times 1} = \vec{A} \cdot \vec{B} \quad (\text{直接使用 NumPy 点积 } \texttt{np.dot(left, right)})$$

无需再做开方与除法运算，CPU 相似度比对性能提升数十倍。

---

## 4. SQLite 持久化与内存二进制转换

SQLite 本身不支持 NumPy 数组对象，只支持 `BLOB`（二进制字节块）。

```text
[写库流程]
文本片段 -> LocalEmbedder.embed_documents() -> np.ndarray (1024维 float32)
         -> vector_array.tobytes() 转换为 4096 字节 raw bytes
         -> 存入 SQLite embeddings 表的 vector (BLOB) 字段

[读库流程]
SQLite vector (BLOB) 字段 -> row["vector"] (4096 字节 raw bytes)
                          -> np.frombuffer(row["vector"], dtype=np.float32) 0拷贝瞬间还原
                          -> np.ndarray (1024维 float32) 进行 np.dot() 计算
```

---

## 5. 数据库 4 表 JOIN 关联结构

检索时通过外键关联链条（`embeddings -> chunks -> extractions -> documents`）拼装出供比对与展示的完整视图：

```text
SELECT d.id, d.path, c.chunk_index, c.text, e.vector
FROM embeddings e
JOIN chunks c       ON c.id = e.chunk_id
JOIN extractions x  ON x.id = c.extraction_id
JOIN documents d    ON d.id = x.document_id
WHERE e.model_id = ?
```

### 关联视图字段说明

| 来源表        | 关键字段              | 作用                                      |
| :------------ | :-------------------- | :---------------------------------------- |
| `documents`   | `path`                | 最终展现给用户的匹配文件路径              |
| `extractions` | `status`              | 过滤仅包含提取成功 (`success`) 的文档     |
| `chunks`      | `chunk_index`, `text` | 匹配到的片段序号与原文字段（供 CLI 预览） |
| `embeddings`  | `model_id`, `vector`  | 锁定模型 ID 并取出二进制向量用于点积计算  |
