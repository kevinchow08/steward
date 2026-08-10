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

---

## 6. 文档级向量加权池化 (Document Vector Weighted Pooling)

当一份长文档包含 $N$ 个切片 Chunk 向量 $(\vec{v}_1, \vec{v}_2, \dots, \vec{v}_N)$ 时，需要合成为代表整份文档全局语义的唯一 $1024$ 维向量 $\vec{V}_{\text{doc}}$。

### 算力与几何步骤：
1. **字符长度加权**：按切片文本字符长度 $w_i = \text{len}(c_i)$ 计算权重比例 $\alpha_i = \frac{w_i}{\sum w_j}$，长段落赋予更高比重。
2. **沿轴降维 (`axis=0`)**：利用 `np.average(vectors, weights=..., axis=0)` 将 2D 矩阵 `(N, 1024)` 沿列方向加权压缩为 1D 向量 `(1024,)`。
3. **L2 范数归一化**：将合成向量除以其模长 `norm = np.linalg.norm(doc_vec)`，重新拉伸为单位球面向量 ($\|\vec{V}_{\text{doc}}\| = 1.0$)，为后续高维余弦点积打下基础。
4. **`float32` 强转**：调用 `.astype(np.float32)` 将计算过程中产生的 `float64` 还原为单精度 32 位浮点数，节省 50% 内存空间。

```python
# 核心 Pooling 实现范式
def compute_document_vector(chunk_vectors, chunk_lengths):
    vectors = np.array(chunk_vectors, dtype=np.float32)  # (N, 1024) 2D 矩阵
    weights = np.array(chunk_lengths, dtype=np.float32)  # (N,) 1D 权重
    
    total_weight = np.sum(weights)
    if total_weight > 0:
        normalized_weights = weights / total_weight
    else:
        # 防御性容错：异常空数据时自动退回均匀平均
        normalized_weights = np.ones(len(weights), dtype=np.float32) / len(weights)
        
    doc_vec = np.average(vectors, weights=normalized_weights, axis=0)
    norm = np.linalg.norm(doc_vec)
    if norm > 0:
        doc_vec = doc_vec / norm
        
    return doc_vec.astype(np.float32)
```

---

## 7. AI 工程师必备的 5 大 NumPy 矩阵与向量核心操作速查

本节梳理了在向量检索、聚类与打标中频繁使用的 5 大底层 NumPy 操作，重点关注 **Shape (形状)** 与 **`axis` 压缩维度** 的物理含义。

```text
       ┌─────────────────────────────────────────────────────────┐
       │ 二维矩阵 (N 行, 1024 列): (N, 1024)                      │
       │  Row 0: [ v_0,0 , v_0,1 , ..., v_0,1023 ]                │
       │  Row 1: [ v_1,0 , v_1,1 , ..., v_1,1023 ]                │
       │  ...                                                    │
       │  Row N: [ v_N,0 , v_N,1 , ..., v_N,1023 ]                │
       └──────────────────────────┬──────────────────────────────┘
                                  │
      axis=0 垂直向下压缩 (沿列求和/均值)  │  axis=1 水平向右压缩 (沿行求和/均值)
                                  ▼
                     一维向量 (1024,): 代表质心/全量均值
```

### 1. `axis=0` 垂直向下滑动降维 (压缩行)
* **算力场景**：求 $N$ 个向量的质心 (Centroid) 或加权池化 (Weighted Pooling)。
* **代码范式**：
  * `centroid = np.mean(c_vecs, axis=0)` —— 输入 `(N, 1024)` 二维矩阵，沿行维度取平均，输出 `(1024,)` 一维质心向量。
  * `doc_vec = np.average(vectors, weights=w, axis=0)` —— 输入 `(N, 1024)` 与 `(N,)` 权重，输出 `(1024,)` 向量。

### 2. 矩阵乘法/点积 `np.dot()` 的 3 种维变模态
在归一化向量场中，`np.dot()` 即在批量计算余弦相似度 ($\cos\theta$)：

* **模态 A (向量 · 向量)**：`(1024,) · (1024,)` $\rightarrow$ 输出 **标量 (Scalar 0.0~1.0)**。依据几何投影物理原理，计算两文档在 1024 维空间里的方向重合能量度。
* **模态 B (矩阵 · 向量)**：`(N, 1024) · (1024,)` $\rightarrow$ 输出 **一维数组 `(N,)`**。一次性算出簇内 $N$ 份文档各自到质心向量的相似度得分列表。
* **模态 C (矩阵 · 矩阵)**：`(N, 1024) · (1024, M)` $\rightarrow$ 输出 **二维打分表 `(N, M)`**。拿 $N$ 份文档分别与 $M$ 个候选标签 Tag 算全量余弦相似度！

> **为什么必须做矩阵转置 (`T.T`)？**
> 初始标签矩阵 `T` 的形状为 `(M, 1024)`（包含 M 个横着躺的标签向量）。由于线性代数矩阵乘法法则要求“左矩阵的列数必等于右矩阵的行数”，且结果的 $(i, j)$ 位置代表“第 $i$ 份文档与第 $j$ 个标签的点积”。
> 必须对标签矩阵求转置 `T.T` 变为 `(1024, M)`，将第 $j$ 个标签向量竖立为第 $j$ 列。如此一来 `D · T.T` 的物理计算 `(N, 1024) · (1024, M) = (N, M)` 正好让文档 $i$ 的行与标签 $j$ 的列精确相乘，符合矩阵乘法的严谨公理，数据绝不会错乱！

### 3. L2 范数算距离与归一化重新拉伸
* **模长算力**：`norm = np.linalg.norm(v)` —— 基于高维勾股定理 $\text{norm} = \sqrt{\sum v_i^2}$ 计算向量箭头的绝对几何长度。
* **归一化投影**：`v_normalized = v / norm` —— 将非单位向量长短缩放（模长大于 1.0 时缩小，小于 1.0 时拉伸），方向保真，尖端重新精准拉回到 $R=1.0$ 的 1024 维单位超球面上。

### 4. 散列 Python List 提升为连续 2D 矩阵
* **代码范式**：`matrix_2d = np.array([v1, v2, v3], dtype=np.float32)`
* **物理含义**：将原本分散存放在 Python 内存堆上的 $M$ 个 1D 向量 `(1024,)`，连续拼接排布成物理连续内存的 `(M, 1024)` 2D 矩阵，解锁 CPU SIMD 并行点积能力。

### 5. 高维相似度排序与 Top-K 索引提取 (`np.argsort`)
* **升序排序索引**：`np.argsort(sims)` —— 不更改 `sims` 数组内部的得分值，仅返回从小到大排序后的原始**下标 Index 索引数组**。
* **降序 Top-K 索引提取**：`np.argsort(-sims)` —— 通过传入负值 `-sims` 实现按相似度从高到低的排序索引，可配合 Python 推导式 `[dids[idx] for idx in sorted_indices[:5]]` 毫秒级提取距离质心最近的前 5 份代表性文档。

---

## 8. 高维数据结构补全：3D 张量与广播机制 (Broadcasting)

### 1. 3D 张量 (3D Tensor) 的物理空间映射
* **1D (向量)**：一条线上的数据，如 `[0.1, 0.2, 0.3]`，Shape 为 `(3,)`。
* **2D (矩阵)**：一个平面的 Excel 表格或单通道灰度图，Shape 为 `(H, W)`。
* **3D (张量)**：将多个 2D 表格像三明治一样叠加为立方体！
  * **彩色 RGB 图像**：由红(Red)、绿(Green)、蓝(Blue) 3 张 `H × W` 灰度灰度图叠在一起，构成 `(Height, Width, 3)` 3D 张量。每一个像素点由 `[R, G, B]` 3 元素列表唯一确定。

### 2. NumPy 广播机制 (Broadcasting) 代码实战
当对维度不匹配的矩阵与向量做元素级加减乘除时，NumPy 会在 C 语言底层自动将低维向量沿缺失轴向拓展复制，避免手动写低效循环。

```python
import numpy as np

# 假设 3 个学生的数学与英语成绩矩阵 (3 行 2 列)
scores = np.array([
    [80, 70],  # 学生 0
    [90, 85],  # 学生 1
    [60, 50]   # 学生 2
], dtype=np.float32)

# 统一调分规则 (1D 向量): 数学 +5 分，英语 +10 分
bonus = np.array([5, 10], dtype=np.float32)  # Shape (2,)

# 触发广播机制 (Broadcasting): NumPy 自动把 bonus 复制扩张成 (3, 2) 矩阵与 scores 对齐按位置相加
final_scores = scores + bonus

print(final_scores)
# 输出:
# [[ 85.  80.]
#  [ 95.  95.]
#  [ 65.  60.]]
```

