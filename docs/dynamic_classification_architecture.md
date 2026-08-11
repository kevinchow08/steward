# 端侧全动态语义分类与打标签引擎架构设计方案 (Phase 3)

本方案旨在为 `steward` 建立 100% 端侧运行、零硬编码规则、高扩展且极致高效的**动态语义分类与打标签引擎**（Dynamic Semantic Classification & Tagging Engine）。

系统核心遵循：**“算法负责发现结构，LLM 负责语义命名，向量矩阵负责海量匹配”** 的分工哲学。

---

## 一、 整体架构与分工哲学

系统将推理重任解耦为三大算力层：

```text
 ┌─────────────────────────────────────────────────────────────┐
 │ 1. 无监督聚类层 (HDBSCAN 算法)                               │
 │    - 负责从高维向量空间自动发现文件的自然高密度归群 (Clusters)    │
 │    - 完全无需预定义分类数量 K，自动识别杂质与孤立点 (Outliers)     │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼─────────────────────────────┐
 │ 2. 端侧轻量 LLM 语义理解层 (Local SLM: Qwen2.5-3B)            │
 │    - 仅对每个 Cluster 抽取 5~10 份代表性文件做语义理解           │
 │    - 产出簇级元数据：主分类名称 (Category) + 候选标签池 (Tag Pool) │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼─────────────────────────────┐
 │ 3. 向量矩阵轻量打标层 (NumPy Matrix Multiplication)          │
 │    - 将 Tag Pool 中的候选词向量化                                │
 │    - 利用 1024 维文档向量与标签向量矩阵做点积 (Dot Product)        │
 │    - 毫秒级算出每个文档专属的 Top-K 标签及置信度                │
 └─────────────────────────────────────────────────────────────┘
```

---

## 二、 六步管道流程 (End-to-End Pipeline)

```mermaid
flowchart TD
    A[Chunk Vectors 块向量] -->|加权池化 Weighted Pooling| B[Document Vector 文档全局向量]
    B -->|HDBSCAN 无监督聚类| C[Semantic Clusters 语义簇]
    C -->|采样代表性文件 5~10 份| D[Local SLM 理解与命名]
    D -->|生成| E[Cluster Category & Tag Pool]
    B & E -->|向量点积 Top-K 匹配| F[Document Category & Specific Tags]
    F -->|模板合成 Zero-LLM| G[Reasoning & Confidence]
    G -->|SQLite 事务| H[SQLite 持久化落盘]
```

### 1. Document Embedding 聚合 (文档级向量合成)
* **输入**：已有 `chunks` 表中的 `1024` 维切片向量。
* **处理**：按字符长度加权平均（Weighted Pooling），直接在内存里合成唯一的 `document_vector` (1024 float32)。
* **优势**：开销仅几毫秒，无需再次调用 `bge-m3` 模型。

### 2. Semantic Cluster 生成 (无监督语义聚类)
* **算法**：采用 **HDBSCAN**（基于密度的层次聚类）。
* **产出**：将文档归纳为 $N$ 个高密度聚类簇，自动识别噪声离群文件（Outliers）。

### 3. Cluster 语义理解与候选标签池 (Local SLM)
* **抽样**：从每个聚类簇中挑选 5~10 份距离簇质心（Centroid）最近的代表性文件。
* **LLM 任务**：调用端侧 `Qwen3.5-2B` 生成：
  ```json
  {
    "category": "财务报销",
    "tag_pool": ["发票", "滴滴", "高铁票", "酒店住宿", "餐饮", "采购", "核销"]
  }
  ```

#### 💡 核心设计规范：算法簇与业务主分类的 N:1 解耦与 SLM 语法收敛控制
1. **N:1 解耦结构**：
   * HDBSCAN 算法层可能会将文档划分为粒度较细的多个离散簇（例如 Cluster 1: 餐饮发票，Cluster 5: 滴滴行程单）；
   * SLM 在理解时，将两者统一归纳收敛到抽象的业务主分类 **`[财务报销]`** (`category_id` 指向同记录)。
   * **价值**：主分类保持极简（宏观不臃肿），具体差异交由细粒度 `tag_pool` 与 `cluster_id` 体现。

2. **三重 SLM 分类收敛控制机制**：
   * **动态分类池记忆 (`existing_categories`)**：处理后置 Cluster 时，Prompt 会自动带入前面已识别的分类名称列表（如 `[财务报销]`），提示 SLM 优先评估并复用，防止分类名称离散碎片化。
   * **强负向约束 (Negative Constraints)**：Prompt 明确规定 `category` 仅允许 2-4 字宏观分类，严禁将“打车、餐饮、Python”等细节词填入 `category`，细节词强制弹压入 `tag_pool`。
   * **离线降级规则归一化**：在无本地 LLM 接入时，启发式规则（`HeuristicFallbackClusterLLM`）亦保持相同的合并分类收敛逻辑。

### 4. 文档主分类与动态 Tag 分配 (矩阵点积 Top-K)
* **Tag Pool 向量化**：调用 `bge-m3` 将 `tag_pool` 中的候选词转化为 $M \times 1024$ 的标签矩阵 $\mathbf{T}$。
* **向量乘法计算**：对每个文档向量 $\mathbf{d}$，计算 $\mathbf{S} = \mathbf{d} \cdot \mathbf{T}^T$，并做 `np.clip(S, 0.0, 1.0)` 防数值溢出。
* **挑选特征标签**：选取得分最高的前 $K$ 个标签作为该文档的专属 Tag（例如：同属“财务报销”类，文件 A 匹配 `["滴滴", "交通"]`，文件 B 匹配 `["酒店", "住宿"]`）。

### 5. 零 LLM 消耗推理理由生成 (Template Reasoning)
* **无需调用 LLM**，通过模板与高维相似度打分快速合成落盘：
  > `"HDBSCAN 向量空间自动分簇 (簇 #1)。质心相似度: 0.88。匹配标签: 餐饮(0.92), 发票(0.85)。"`

---

## 三、 SQLite 数据表扩展设计

除了已有的 `documents`、`categories` 和 `tags` 表，新增 **`semantic_clusters`** 语义簇表：

```sql
/* 1. 语义簇表：保存 AI 发现的聚类结构与 Tag Pool (category_id 为 ON DELETE SET NULL 解耦外键) */
CREATE TABLE IF NOT EXISTS semantic_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id INTEGER,
    centroid_vector BLOB NOT NULL,       -- 簇中心向量 (1024D float32)
    tag_pool_json TEXT NOT NULL,         -- LLM 生成的候选标签池 JSON 数组
    summary TEXT,                        -- 簇级内容摘要
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL
);

/* 2. documents 表扩展：新增簇关联 */
ALTER TABLE documents ADD COLUMN cluster_id INTEGER REFERENCES semantic_clusters(id);
```

---

## 四、 方案总结与核心优势

1. **解决性能痛点**：对于 10,000 个文档，LLM **仅需推理 10~20 次（对聚类簇）**，整体分类耗时从 30 分钟骤降至 10 秒以内！
2. **解决分类质量**：主分类保持归纳抽象（`Category` N:1 收敛），个体标签体现差异化细节（`Tags`），既有秩序又保留个性。
3. **零云端依赖**：全流程依赖 `numpy` + `HDBSCAN` + 端侧 3B SLM，100% 本地隐私安全。

