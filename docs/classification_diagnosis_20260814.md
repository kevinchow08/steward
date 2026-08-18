# Cluster-then-Label 管线现状诊断（2026-08-14）

本文档基于对代码的实际通读（而非文档/注释）产出，目的是在讨论"动态无监督分类"下一步之前，先对齐现状、验证已知痛点、揪出补丁式代码，并列出需要用户澄清的目标性问题。

---

## 一、端到端数据流：代码对应关系

实际入口是 `main.py classify` → `run_classify()` → `semantic_classifier.run_dynamic_classification_pipeline()`（[main.py:127](../main.py#L127)、[semantic_classifier.py:119](../src/steward/semantic_classifier.py#L119)）。管线按以下顺序串联：

| 阶段 | 文件/函数 | 说明 |
|---|---|---|
| 0. 清空旧数据 | `document_index.DocumentIndex.clear_classifications()` | 每次重跑先删光 `document_tags` / `document_classifications` / `semantic_clusters` / `tags` / `categories`，`documents.cluster_id` 置空 |
| 1. 文档级向量池化 | `document_vectors.get_all_document_vectors()` → `compute_document_vector()` | 按 chunk 字符长度加权平均 + L2 归一化，把一个文档的所有 chunk 向量合成一个 1024D 向量 |
| 2. 聚类 | `clustering.cluster_document_vectors()` | UMAP(1024D→20D，cosine) 降维 → HDBSCAN（euclidean, `leaf`）聚类 |
| 2.5 边缘归属 | `semantic_classifier.py:163-195`（管线函数内联，无独立模块） | 对 HDBSCAN 判为 Outlier 的文档，在原始 1024D 空间找最近簇质心，相似度 ≥0.50 才收编，否则留作 Outlier |
| 3. 簇命名 | `cluster_llm.analyze_clusters_metadata()` → `LocalHttpClusterLLM.generate_cluster_metadata()` | 对每簇的 5 份代表文档调用本地 SLM（llama.cpp server, `localhost:8080/v1`），产出 `category` + `tag_pool` |
| 3.5 Taxonomy 归一化 | `cluster_llm._normalize_taxonomy()` | 收集所有簇产出的 category 名，再调一次 SLM 合并同义词 |
| 4. 打标签 | `tag_matcher.match_tags_for_documents()` | 全部候选 tag 向量化后与全部文档向量做 2D 矩阵点积，`score ≥ 0.55` 即命中 |
| 5. 落盘 | `semantic_classifier.py:215-272` 直接调用 `DocumentIndex.save_classification()` / `save_semantic_cluster()` | 写入 SQLite 四张表 |

这条链路和你描述的 "Cluster-then-Label" 架构**一致**：向量化 → 降维聚类 → 只对 K 个簇调 SLM → 矩阵点积分配标签，全程只在第 3 步调用 LLM，是 O(K) 而不是 O(N)。

**旁支（未接入主链路，需要你决定去留）**：`semantic_classifier.classify_document_text()` 是一套纯关键词硬编码分类器（见下文第三节），只被 `tagging.py:build_classifications()` 调用，而 `build_classifications()` 在整个仓库里没有任何调用方——`main.py` 没有任何子命令指向它。这是 Week 3 早期（commit `e5e41c9`）留下的旧实现，被后来的 HDBSCAN+SLM 管线（`d73189f` 之后）替代，但代码没有删除。

---

## 二、已知痛点验证结果

### 痛点 1："简历因关键词与技术教程重合被错误聚类"

**未验证**。当前数据库里跑的语料（`~/Documents/doc/对话List` 等目录，482 份文档，见 `output/tags_report.md`）里没有找到任何简历类文件（grep "简历/resume/求职" 无命中于实际分类结果）。这条痛点目前只存在于此前的架构讨论中，代码里没有专门处理简历的逻辑，也没有实测证据。如果要验证，需要往语料里补充真实简历文件后重跑 `classify`。

### 痛点 2："聊天记录因结构相似被聚成一坨，混入的理财话题导致整簇被错误命名"

**部分验证，且实测发现了一个更具体的表现形式**。在 `output/tags_report.md` 里找到两个直接证据：

- 第 98 条：`与Gemini聊关于m-flo的话题.txt`（聊音乐话题的对话记录）被分进 **[投资理财]** 簇（簇 #28），置信度 0.89，唯一匹配的标签是"满足感(0.56)"——明显是聊天记录被拖进了不相关的簇。
- 第 99 条：`与Claude关于distyclean的对话1.txt`（内容是纯技术讨论：API/数据库/前端/后端）被分进 **[财务报销]**（簇 #10），置信度 0.94，但打上的 179 个标签里绝大多数是"前端/后端/API/数据库"这类技术词，只有零星几个"发票/报销"相关词。

这两条都指向同一个根因：`对话List` 目录下的聊天记录文本结构相似、话题跨度大，HDBSCAN 在向量空间里容易把它们和某个偶然邻近的簇（哪怕主题不符）聚在一起，然后 SLM 只看 5 份代表文档命名整簇，命名结果对整簇（包括跑偏的边缘文档）通吃。这与你描述的痛点 2 机制一致，只是触发对象不局限于"理财话题"，任何小众话题的聊天记录都可能被顺带卷入。

### 新发现（不在你列出的痛点里，但代码证据确凿）：Tag Pool 是全局共享的，不按簇/分类隔离

`tag_matcher.match_tags_for_documents()`（[tag_matcher.py:44-50](../src/steward/tag_matcher.py#L44)）把**所有簇**产出的 `tag_pool` 合并去重成一个全局标签表，再让**每一份文档**去和这个全局表里的每个标签做点积比对，`score ≥ 0.55` 就算命中。

后果：设计文档（`docs/dynamic_classification_architecture.md` 第 82 行）描述的效果是"同属财务报销类，文件 A 匹配滴滴/交通，文件 B 匹配酒店/住宿"——即标签应该在同一分类内做区分。但实际实现里标签匹配跟文档所属的簇/分类完全无关，只要向量点积分数够高就命中，导致内容跨度大的文档（尤其是聊天记录）会被打上上百个跟它实际分类毫不相关的标签（上面第 99/102/104/106 条记录，单文档标签数普遍 80~180 个）。这是当前"标签质量"问题的直接技术原因，比"聚类不准"更容易独立修复（调高阈值、或把候选标签范围限制在文档所属簇的 tag_pool 内，而不是全局池）。

---

## 三、已确认技术细节核对

| 你确认的细节 | 代码实际情况 | 结论 |
|---|---|---|
| HDBSCAN `cluster_selection_method='leaf'` | [clustering.py:91](../src/steward/clustering.py#L91) `cluster_selection_method="leaf"` | ✅ 一致 |
| 本地 SLM 用 `Qwen3.5-2B-Q4_K_M.gguf`（llama.cpp） | [cluster_llm.py:36](../src/steward/cluster_llm.py#L36) 默认 `model_name="Qwen3.5-2B-Q4_K_M.gguf"`，走标准 OpenAI SDK 打 `localhost:8080/v1` | ✅ 一致（但注意 `docs/dynamic_classification_architecture.md` 第 21 行文档里写的是过时的 "Qwen2.5-3B"，文档没跟着代码更新） |

**其他参数（你没提到，但值得记录）：**

- `min_cluster_size`：实际管线调用是 `3`（[semantic_classifier.py:123](../src/steward/semantic_classifier.py#L123) 函数签名默认值），**不是** `clustering.py` 里自己声明的默认值 `5`（[clustering.py:42](../src/steward/clustering.py#L42)）。因为 `semantic_classifier.py` 调用 `cluster_document_vectors()` 时显式传了 `min_cluster_size=min_cluster_size`（=3），覆盖了 `clustering.py` 的默认值。两处默认值不一致容易造成"看代码以为是 5，实际跑的是 3"的误解。
- `min_samples`：实际用的是 `clustering.py` 的默认值 `2`（管线没有覆盖它）。
- `umap_n_components=20`，`umap_n_neighbors=30`（管线未覆盖，用的是 `clustering.py` 默认值）。
- Tag 匹配相似度阈值 `tag_threshold=0.55`（[semantic_classifier.py:124](../src/steward/semantic_classifier.py#L124)）。
- Outlier 最近质心兜底阈值 `0.50`（[semantic_classifier.py:188](../src/steward/semantic_classifier.py#L188)，硬编码在管线函数体内，不是可配置参数）。

---

## 四、"unclassified 兜底"原则：CLAUDE.md 里写的规则，实际管线没有实现

这是本次诊断里最值得你注意的一条。CLAUDE.md 的 Week 3 目标写明"当置信度低于阈值（如 `< 0.70`）时，如实标为未归类"。代码里确实存在这个常量——`semantic_classifier.py:13` 定义了 `CONFIDENCE_THRESHOLD = 0.70`——但它**只在旧的、未被调用的 `classify_document_text()`（第二节提到的旁支）里生效**。

`main.py classify` 实际跑的 `run_dynamic_classification_pipeline()` 判定 `unclassified` 的逻辑是（[semantic_classifier.py:265](../src/steward/semantic_classifier.py#L265)）：

```python
status="classified" if match.category != "未分类" else "unclassified"
```

而 `match.category` 只有在文档**没有被任何簇收编**（HDBSCAN 判为 Outlier 且最近质心兜底相似度也 < 0.50）时才会是 `"未分类"`。也就是说，现在的"是否标为未归类"完全取决于**这份文档是否属于某个簇**，跟文档自己的分类置信度数值（`confidence = 文档向量与簇质心的余弦相似度`）没有任何比较关系——哪怕一份文档和它所属簇质心的相似度只有 0.51，只要它被 HDBSCAN/最近质心分进了某个簇，就会被标记为 `classified`，不会因为置信度低而回退。

换句话说：置信度这个数字目前只是"算出来存进数据库供人看"，并没有真正参与"要不要判定为 unclassified"的决策。这跟 CLAUDE.md 里"低置信度如实标为未归类"的既定原则有实质出入，建议你决定是要把 0.70 阈值判断补回主链路，还是重新定义"未归类"该由什么信号触发（比如簇内一致性、还是文档与质心相似度、还是两者结合）。

---

## 五、补丁式代码清单（硬编码规则/特判逻辑）

repo 里没有 TODO/FIXME/HACK 注释，但 grep 关键词发现两处实打实的硬编码规则表：

1. **`semantic_classifier.py:52-85`** —— `classify_document_text()` 里 5 条基于关键词的 if/elif 规则（简历/财务/技术文档/周报/证件合同），命中就给固定的 category + 固定置信度（0.85~0.95）。**属于死代码**（见第一节），不影响当前 `classify` 命令的实际行为，但如果以后有人不小心把 `tagging.py` 接回 `main.py`，这套硬编码规则会原样复活。建议要么明确删除，要么在文件顶部注明"未接入主链路，仅供参考"。

2. **`cluster_llm.py:157-191`** —— `HeuristicFallbackClusterLLM`，同样是关键词规则表（财务报销/招聘简历/技术代码/工作周报四类 + 兜底"综合文档"）。**这处是活代码**：当本地 llama.cpp server 连不上或推理报错时（[cluster_llm.py:151-154](../src/steward/cluster_llm.py#L151)），管线会静默降级到这套硬编码规则，不会中断。风险在于：这套规则表是针对你当前语料手调的（简历/财务/技术/周报四类），换一个用户或换一个目录，SLM 服务只要没启动，所有簇都会被这四类规则或"综合文档"兜底分类，体验上会看起来像是"分类引擎在正常工作"，但实际上是关键词规则在冒充语义分类结果，不会有任何日志/提示告诉用户"这次分类退化成规则匹配了"。这是最值得关注的一处补丁——它换目录会失效，且失效方式是静默的。

此外 `cluster_llm.py` 的 SLM Prompt 本身（[cluster_llm.py:109-124](../src/steward/cluster_llm.py#L109)）里塞了几个具体的 few-shot 例子（"报销相关"目录→财务报销、"Nest 通关秘籍"目录→技术代码），这是提示词工程意义上的引导，不是代码分支意义上的硬编码，但如果换语料后这几个例子的措辞对新领域没有代表性，可能需要跟着调整。

---

## 六、其他代码层面的小问题（顺手发现）

- `requirements.txt` 缺失实际在用的依赖：`scikit-learn`（HDBSCAN 来源）、`umap-learn`、`openai`、`httpx` 都已装在 `.venv` 里且被代码直接 import，但没写进 `requirements.txt`。换一台机器 `pip install -r requirements.txt` 会直接跑不起来。
- `docs/dynamic_classification_architecture.md` 里模型名写的是 "Qwen2.5-3B"，代码实际是 "Qwen3.5-2B"，文档没跟上代码改动。

---

## 七、终局目标：目前代码/文档里看不出答案，需要你回答

通读全部代码和三份现有文档后，以下问题没有找到任何明确指向，不替你假设，列出来由你决定：

1. **多模态扩展到什么程度？** CLAUDE.md 提到"从纯文本扩展到图片、视频"是长期方向，但现在连"先做图片 EXIF/文件名，还是先做图片内容 OCR/视觉语义"这种选型都没有定论。是否需要现在就定一个 Week 4/5 的候选顺序？
2. **要不要支持自动移动/重命名文件，只是打标签+索引？** CLAUDE.md 产品化方向里第二层（移动归档、重命名）明确写了"建议先提议+用户确认"，但没有一个决策：这个"提议+确认"的产品形态具体长什么样（CLI 交互式确认？生成一份 diff 报告让用户审阅？）现在完全没有设计，值得提前想一下大概方向，即使不实现。
3. **准确率的可接受下限是多少？** 目前"分类是否正确"完全靠人工翻 `tags_report.md` 判断，没有任何量化指标（比如抽样人工标注 100 份、算准确率）。要不要建立哪怕最简陋的评估集，用来判断"这周改的参数/prompt 到底是变好了还是变坏了"？现在每次调参（比如 `min_cluster_size` 从 5 改到 3）都是靠肉眼看结果是否"看起来合理"，没有可比较的基线。
4. **换用户/换目录的"合格"泛化能力怎么定义？** 第五节提到的 `HeuristicFallbackClusterLLM` 硬编码规则表是针对当前语料手调的最明显例子——如果这套系统要具备"面向新用户开箱可用"的泛化能力，现在的启发式降级路径本身就是一个反例（换目录后要么失效要么误导）。这跟"离线兜底策略应该是什么样"是一个具体可以现在就讨论的子问题：SLM 服务连不上时，到底应该（a）像现在这样静默退化成关键词规则，（b）明确报错/在结果里标注"本次为降级结果"，还是（c）把整个 pipeline 直接失败并提示用户先启动 SLM 服务？
5. **Tag Pool 是否应该按簇/分类隔离？** 见第二节新发现的问题——这个决定会实质性改变标签质量，值得先确认设计意图（"标签本来就该是全局共享的、任何分类都可能出现同一个标签"，还是"标签应该只在其所属簇/分类内比较，避免噪音"），再决定怎么改 `tag_matcher.py`。
6. **`unclassified` 判定要不要把置信度阈值补回来？** 见第四节，这是 CLAUDE.md 既定原则和当前代码实现之间最直接的出入，需要你确认是要修代码对齐文档，还是修文档对齐代码（重新定义"未归类"该基于什么信号）。
