# 检索质量优化 · 交接说明

这份文档是给负责"检索结果质量优化"的新 session 的。**只写事实性背景和协作约定，故意不写任何关于"检索哪里有问题、什么方向试过/没用"的结论。** 原因见第 1 节。

## 1. 先读这一条：请自己从头测，不要被旧记录带偏

- 这个项目之前做过很多检索测试。那些记录是当时的人（含 AI）在特定语料、特定判断标准下留下的，**没有经过独立验证，可能有偏差，也可能已经过时**。
- 为了避免锚定，`docs/product_direction_and_roadmap.md` 里原有的检索测试记录和结论**已经被移除**（地基审计表里"语义搜索"一行只留了指向本文档的说明）。移除的内容可以从 git 历史找回，但**请先不要去翻**。
- 仍然留在仓库里、会锚定你的东西，**请先不要读**：
  - `tests/search_regression_cases.json` 和 `tests/run_search_regression.py`：旧的回归案例集（含故意保留的"已知失败"案例和各案例的预期判断）。
  - git 历史里与检索相关的提交信息和旧版 roadmap。
  - `docs/week2_document_semantic_index_plan.md` 属于早期设计背景，可以读，但里面的设想不等于现状。
- 项目根目录的 `CLAUDE.md` 会引导你读 `docs/` 下的几份文档，对这个任务可以读，检索部分已清理。
- 正确的顺序：先自己用真实语料测，自己发现问题、形成判断；**形成判断之后**再去读上面那些旧记录，把它们当作"待验证的假设"做对照，而不是当作结论。
- 如果你的发现和旧记录矛盾，以你自己的、可复现的实测为准，并把矛盾明确写出来。

## 2. 你的任务

"现在的检索结果离真实用户可用还有差距，不够智能。"根本问题在哪、怎么改，**由你自己测出来再决定**，没有预设方案。

建议的起手方式（不是强制）：

1. 搞清楚现在检索是怎么工作的（看代码，自己跑）。
2. 建一套**固定的评测集**（查询 + 哪些文件算正确答案 + 查询类型）和指标，再谈优化。没有它，没法判断一次改动是变好还是变坏。**标注（"这条查询我想找的是哪个文件"）只有项目主人能给**，需要时直接向他要，起步 30–50 条真实查询即可。
3. 评测集分"调参集"和"留出集"，防止只对眼前案例有效。

## 3. 系统背景（事实）

- **是什么：** 本地运行的文件语义索引/打标签/检索工具（macOS，Python，CLI 名 `steward`）。端侧运行，不依赖云端 API。
- **环境：** MacBook Pro M4 Pro，48GB 统一内存。虚拟环境 `.venv`，已 `pip install -e .`。
- **检索入口：** `steward search "<查询>" [--top-k N] [--candidates N]`。核心代码 `src/steward/semantic_search.py`、`src/steward/reranker.py`；文本切片 `chunking.py`；向量 `embeddings.py`、`document_vectors.py`；库操作 `document_index.py`。
- **模型：** 本地 embedding（BAAI/bge-m3）和 reranker（bge-reranker-v2-m3），放在 `~/Library/Application Support/Steward/models/`。打标签/理解用的 LLM 是本机 llama-server（OpenAI 兼容接口，默认地址 `http://127.0.0.1:8080/v1`，定义在 `tagging.py` 的 `DEFAULT_LLM_BASE_URL`），不一定在线，用前先 `curl` 它的 `/health` 确认。
- **数据：** SQLite 库在 `~/Library/Application Support/Steward/steward.db`（约 1 GB），内含文档元数据、文本 chunk、向量、标签。语料是项目主人的**真实个人文件**（含隐私内容），不要把内容发到任何外部服务。
- **⚠️ 真实库会自己变。** 本机有 launchd 任务（`com.steward.refresh`）会在文件变化或每 3 小时自动增量刷新这个库。做评测前请**复制一份库快照**（例如复制到 scratch 目录）并指向它（CLI 有 `--db` 参数），否则语料一变，前后分数没法比较。
- **不要对真实库跑会写入的命令**（`index`、`tag`、`refresh`、`duplicates` 等），除非先确认。检索本身是只读的。

## 4. 协作边界

| | 归属 |
|---|---|
| 你负责 | `semantic_search.py`、`reranker.py`、`chunking.py`、`embeddings.py` 等检索相关代码；你新建的评测脚本和评测集 |
| 主线 session 负责 | `refresh.py`、`tagging.py`、`duplicates.py`、索引/扫描流程、launchd 相关 |
| 需要事先沟通 | 改 `document_index.py` 的表结构（会影响主线的写入逻辑）；新增会被主线调用的接口 |

- **用单独的分支或 worktree 开发**，不要直接在 `main` 上改。合并后项目主人会让两边 session 互相 review。
- 项目偏好（来自 CLAUDE.md，请遵守）：一步一步来、先讨论方案再动手；不要过度设计和不必要的抽象；代码里多写解释性注释（项目主人 Python 不是主力语言）；遇到问题报告真实报错。

## 5. 结果怎么反馈

- 结论按日期写进**独立的新文件** `docs/retrieval_optimization_log.md`（不要写回 roadmap，避免两边记录混在一起）。
- 每条记录建议包含：做了什么实验、评测集上的数字（前后对比）、结论、是否建议合入。
- 无论成功还是失败（"试了没用"也是有价值的结论）都要记。主线 session 会读这个文件来接上进展。
