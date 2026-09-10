"""CLI 入口:子命令形式（index / search / tag / tags / duplicates），串联
scan -> index -> tag -> search/duplicates 这条链路。

这个模块现在住在包内部（src/steward/cli.py），不是仓库根目录的 main.py——
所以不需要 sys.path 技巧就能 import steward.*，pip 装到哪都能正常跑。
pyproject.toml 里的入口点 `steward = "steward.cli:main"` 指向下面的 main()。
仓库根目录还留了一个很薄的 main.py，只是为了开发时 `python main.py ...`
还能直接用，真正的逻辑全在这里。
"""

import argparse
import os
from pathlib import Path

from steward.document_index import DEFAULT_DB_PATH
from steward.paths import OUTPUT_DIR


def run_index(target_dir, db_path, force=False):
    """加载本地模型并为目录建立 document 内容索引。

    force：默认 False，走增量。传 True 强制对所有文件重新提取/分段/向量化，
    忽略"内容没变就跳过"的判断——每当处理逻辑本身变了（比如这次新加的
    chunks_fts 关键词索引，老数据在增量判断下永远补不上），需要这个开关，
    跟 tag 命令的 --force 是同一个用途。
    """

    from steward import indexing
    from steward.embeddings import LocalEmbedder

    print(f"正在加载本地 embedding 模型，首次运行可能需要下载模型文件...{'（force 模式：全部重新处理）' if force else ''}")
    embedder = LocalEmbedder()

    try:
        stats = indexing.build_index(target_dir, embedder, db_path=db_path, force=force)
    except FileNotFoundError as e:
        # 目标目录不可达（比如外置盘没连接）——不打印一堆 traceback 吓用户，
        # 老实说清楚原因就够了。
        print(f"❌ {e}")
        return

    print(f"扫描文件: {stats['scanned_files']}")
    print(f"识别到代码项目: {stats['project_count']}（已整体登记，跳过其内部文件）")
    print(f"已建立索引: {stats['indexed_files']}")
    print(f"成功提取: {stats['success_files']}")
    print(f"无文字内容: {stats['no_text_files']}")
    print(f"提取失败: {stats['error_files']}")
    print(f"暂不支持: {stats['unsupported_files']}")
    print(f"跳过未变化文件: {stats['skipped_unchanged_files']}（增量索引生效，内容没变不重新处理）")
    print(f"标记为已消失: {stats['removed_files']}（这次没扫到，可能是删除/改名）")
    print(f"文本片段: {stats['chunk_count']}")
    print(f"索引耗时: {stats['elapsed_seconds']:.2f} 秒")
    print(f"数据库: {db_path}")


def run_search(query, db_path, top_k, candidate_pool_size):
    """加载本地模型，并在已有 document 索引中搜索，同时打印耗时与统计数据。

    两阶段检索：向量+关键词+标签三路粗筛出一批候选，交给本地 cross-encoder
    重排序模型（bge-reranker-v2-m3）做最终精排，详见 semantic_search.py
    的说明。粗筛阶段不再对稠密检索的原始相似度做硬性门槛过滤——"够不够
    相关"完全交给重排序模型判断，这是撞到"银行流水"这个真实反例（向量
    检索把一份不相关的 SQL migration 教程排到比真正相关的发票还高）之后
    改的，重排序模型能正确分辨这种情况（实测那份教程被打到 -10.9 分，
    真正相关的内容在 -2.6~+0.5 分之间）。
    """

    import time
    from steward import semantic_search
    from steward.embeddings import LocalEmbedder
    from steward.reranker import LocalReranker

    total_start = time.monotonic()

    # 1. 测量本地模型加载耗时——embedding 模型和重排序模型是两个独立的
    # 本地模型，各自单独计时，方便看清楚耗时分别花在哪一边。
    print("正在加载本地 embedding 模型...")
    t_model_start = time.monotonic()
    embedder = LocalEmbedder()
    embed_model_load_seconds = time.monotonic() - t_model_start

    print("正在加载本地重排序模型...")
    t_reranker_start = time.monotonic()
    reranker = LocalReranker()
    rerank_model_load_seconds = time.monotonic() - t_reranker_start

    # 2. 执行语义搜索，接收结果和详细耗时/计数
    results, stats = semantic_search.search_documents(
        query,
        embedder,
        reranker,
        db_path=db_path,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
    )

    total_seconds = time.monotonic() - total_start

    if not results:
        # 三路粗筛全部没有候选，或者候选交给重排序之后全部低于相关度门槛，
        # 如实说没找到，不强行凑出 top_k 个看起来正常、实际不相关的结果。
        print("没有找到足够相关的内容。")
    else:
        for index, result in enumerate(results, start=1):
            # 这个 score 是重排序模型给出的原始相关性分数，不是 0~1 的
            # 相似度百分比，也不再是 RRF 融合分数——数值大致在 -11~+1 这个
            # 区间，越高越相关，不要按"匹配度多少%"去解读。
            print(f"{index}. 重排序分数={result.score:.4f}")
            print(f"   文件: {result.path}")
            print(f"   chunk: {result.chunk_index}")
            print(f"   片段: {result.text}")
        # document_count 是候选池（candidate_pool_size 个）里经过重排序、
        # 通过相关度门槛的数量，不是"全语料里有多少真正相关"——粗筛阶段
        # 已经把候选面限制在了候选池大小以内，语料里可能还有没进入候选池、
        # 因此没被重排序看到的相关文档，想要不受候选池上限影响的完整列表，
        # 用 `tags --tag` 按标签穷举查。
        if stats["document_count"] > len(results):
            print(f"（候选池 {stats['candidate_pool_size']} 个里，{stats['document_count']} 个通过重排序的"
                  f"相关度判断，这里只展示前 {len(results)} 个，加大 --top-k 看更多；"
                  "如果想要不受候选池上限影响的完整列表，用 tags --tag 按标签穷举查）")

    # 3. 打印性能与耗时监控信息
    print("-" * 50)
    print("【性能与耗时统计】")
    print(f"候选池大小: {stats['candidate_pool_size']} 个 | 通过重排序: {stats['document_count']} 个 "
          f"| 比较片段(chunks): {stats['chunk_count']} 个")
    print(f"向量检索命中: {stats['dense_hit_count']} 个文档 | 关键词检索命中: {stats['sparse_hit_count']} 个文档 "
          f"| 标签检索命中: {stats['tag_hit_count']} 个文档")
    print(f"embedding 模型加载耗时: {embed_model_load_seconds:.3f} 秒")
    print(f"重排序模型加载耗时: {rerank_model_load_seconds:.3f} 秒")
    print(f"Query 向量化: {stats['query_embed_seconds']:.3f} 秒")
    print(f"向量检索耗时: {stats['vector_search_seconds']:.3f} 秒")
    print(f"关键词检索耗时: {stats['keyword_search_seconds']:.3f} 秒")
    print(f"标签检索耗时: {stats['tag_search_seconds']:.3f} 秒")
    print(f"重排序耗时: {stats['rerank_seconds']:.3f} 秒")
    print(f"搜索总耗时:   {total_seconds:.3f} 秒")


def run_tag(db_path, max_workers=8, force=False, llm_base_url=None):
    """为已有索引文本的文档批量执行打标签（开放式 reasoning + tags，不维护分类体系），并持久化。

    这个函数以前还接受 structural_base_url/structural_model 两个参数，是给 V3
    "taxonomy 归纳"那一步单独指定一个更大模型用的。V3 整套分类体系已经拿掉（见
    docs/dynamic_classification_architecture.md 的"V3 复盘"），现在只剩一种任务
    （逐文件打标签），不再需要区分"抽象归纳"和"具体判断"两种模型，这两个参数
    跟着一起删掉了。

    force：默认 False，走增量（内容没变、已经打过标签的文档跳过，见
    run_tagging_pipeline 的说明）。改了 prompt/snippet 长度/模型这类"打标签
    逻辑本身"的改动之后，传 True 强制全部重新打一遍——增量判断只看内容变没变，
    看不出代码变没变，这种情况必须自己记得手动加这个参数。

    llm_base_url：打标签要调用的 llama-server 地址。None（默认）时用
    run_tagging_pipeline 自己的默认值（本机 127.0.0.1:8080）。分发给别的
    同事用时，他们机器上没跑 llama-server，要指向一台在跑的机器——通过
    `--llm-base-url` 参数或 `STEWARD_LLM_BASE_URL` 环境变量传进来。
    """

    import time
    from steward.document_index import DocumentIndex
    from steward.tagging import run_tagging_pipeline

    print(f"🚀 启动打标签引擎（开放式 reasoning + tags{'，force 模式：全部重新打标签' if force else ''}）...")
    if llm_base_url:
        print(f"   打标签调用的 llama-server: {llm_base_url}")
    start_time = time.monotonic()

    pipeline_kwargs = {"max_workers": max_workers, "force": force}
    if llm_base_url:
        pipeline_kwargs["llm_base_url"] = llm_base_url

    with DocumentIndex(db_path) as index:
        stats = run_tagging_pipeline(index=index, **pipeline_kwargs)

    elapsed = time.monotonic() - start_time

    print("-" * 50)
    print("【逐文件打标签统计】")
    print(f"分析文档总数: {stats['total_documents']} 份")
    print(f"内容没变跳过: {stats['skipped_up_to_date_count']} 份（增量生效，未传 --force）")
    print(f"内容过短跳过: {stats['skipped_short_count']} 份")
    print(f"SLM 解析失败: {stats['parse_failed_count']} 份")
    print(f"深度打标签: {stats['tagged_count']} 份")
    print(f"基础类型识别（非文档类型/未成功提取文本）: {stats['basic_count']} 份")
    print(f"未打标签 (untagged): {stats['untagged_count']} 份")
    print(f"代码项目: {stats['project_count']} 个（{stats['project_tagged_count']} 个已打标签，"
          f"{stats['project_failed_count']} 个失败）")
    print(f"全管线总耗时: {elapsed:.3f} 秒")
    print(f"steward 自身进程 峰值内存: {_format_bytes(stats['steward_peak_rss_bytes'])} | 峰值 CPU: {stats['steward_peak_cpu_percent']:.1f}%")
    if stats["llama_server_peak_rss_bytes"] is not None:
        print(f"llama-server 进程 峰值内存: {_format_bytes(stats['llama_server_peak_rss_bytes'])} | "
              f"峰值 CPU: {stats['llama_server_peak_cpu_percent']:.1f}%")
    else:
        print("llama-server 进程: 没找到，跳过这一路监控（GPU 算力占用暂不测，见 monitor.py 的注释）")
    print(f"数据库持久化: {db_path}")



def run_tags(db_path, tag_query=None):
    """展示 SQLite 中已打标签文档的标签、reasoning 及置信度，并输出到文件。

    tag_query：可选。传了的话不生成全量报告，改成"给我列出所有标签匹配这个
    关键词的文档"——这是跟 search 命令不一样的能力：search 是排序找最相关
    的前几个（会被 top_k 截断、会被相关度门槛过滤，不保证完整），这里是
    穷举式的集合筛选，只要标签匹配就一定会出现在报告里，不会因为其他信号
    弱就被漏掉。底层直接用 search_document_tags()（标签关键词检索，不需要
    加载 embedding 模型，比 search 命令快得多），不经过 RRF 融合排序。
    """

    from steward.document_index import DocumentIndex

    bm25_by_id = None
    with DocumentIndex(db_path) as index:
        if tag_query:
            hits = index.search_document_tags(tag_query)
            if not hits:
                print(f"没有找到标签匹配 {tag_query!r} 的文档。")
                return
            # bm25 排好序的命中列表——记下每个 document_id 的分数和顺序，
            # 待会按这个顺序展示，不要被 iter_tagging_results() 内部按
            # created_at 排序打乱。
            bm25_by_id = {row["document_id"]: row["bm25_score"] for row in hits}
            ordered_ids = [row["document_id"] for row in hits]
            records_by_id = {r["document_id"]: r for r in index.iter_tagging_results(document_ids=ordered_ids)}
            records = [records_by_id[doc_id] for doc_id in ordered_ids if doc_id in records_by_id]
        else:
            records = list(index.iter_tagging_results())

    if not records:
        print("当前没有任何已打标签的文档。请先运行: python main.py tag")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / ("tags_report_filtered.md" if tag_query else "tags_report.md")

    # 三种状态分开统计，报告里也分开展示——"深度打标签"和"基础类型识别"的可信程度
    # 不一样，混在一起看容易把后者的粗糙标签误当成前者那种经过语义判断的结果。
    status_icon = {"tagged": "✅", "basic": "🔹", "untagged": "⚠️"}
    status_label = {"tagged": "深度打标签", "basic": "基础类型识别", "untagged": "未打标签"}
    counts = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print(f"共查找到 {len(records)} 份文档。正在生成报告...")

    with open(report_path, "w", encoding="utf-8") as f:
        if tag_query:
            f.write(f"# 标签筛选报告：{tag_query}\n\n")
        else:
            f.write("# 文档打标签报告\n\n")
        f.write(f"**总计**: {len(records)} 份文档")
        f.write("（" + "，".join(f"{status_label.get(s, s)} {n} 份" for s, n in counts.items()) + "）\n\n")
        for index, r in enumerate(records, start=1):
            tags_str = ", ".join(r["tags"]) if r["tags"] else "(无标签)"
            icon = status_icon.get(r["status"], "❔")
            f.write(f"### {index}. {icon} {status_label.get(r['status'], r['status'])}\n")
            if bm25_by_id is not None:
                # bm25 越小代表匹配度越高，跟 search 命令关键词那一路是同一个
                # 约定，见 semantic_search.py 里的注释。
                f.write(f"- **标签匹配度(bm25，越小越相关)**: {bm25_by_id[r['document_id']]:.4f}\n")
            f.write(f"- **标签**: {tags_str}\n")
            f.write(f"- **置信度**: {r['confidence']:.2f}\n")
            f.write(f"- **文件**: `{r['path']}`\n")
            f.write(f"- **依据**: {r['reasoning']}\n\n")

    print(f"✅ 报告已生成: {report_path}")


def _format_bytes(n):
    """把字节数格式化成人类好读的单位，只用来展示，不参与任何判断逻辑。"""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{n}B" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _report_tag(target_dir, db_path):
    """把这次 duplicates 扫的是什么，变成一段能安全嵌进文件名的标识，用来
    区分不同扫描各自的报告——原来 duplicates_report.md 这类文件名是写死
    的，扫完 ~/Downloads 再扫 ~/Documents，第二次会把第一次的报告覆盖掉，
    这是真实 bug，不是"就该覆盖"的预期行为（预期行为是"同一个来源重复
    跑，新报告覆盖旧报告"，不同来源不该互相覆盖）。

    传了 target_dir：按目录区分。只用目录名（比如"Downloads"）当前缀不
    够——两个不同路径下都可能有一个叫"Downloads"的文件夹，会撞车。所以在
    人类可读的目录名后面缀一段基于完整解析路径算出来的短哈希，既保留
    可读性，又能保证不同目录不会撞到同一个文件名。

    没传 target_dir（数据库全量模式，见 find_duplicates_from_index()）：
    同样的"可读前缀 + 路径哈希"套路，只是换成基于数据库路径区分，因为
    这次的候选集不是某一个目录，是"这个数据库里登记过的全部文件"。
    """
    import hashlib

    if target_dir is not None:
        resolved = Path(target_dir).expanduser().resolve()
        readable = resolved.name or "root"  # 根目录（比如"/"）本身没有 name，兜底一个字符串
        digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
        return f"{readable}_{digest}"

    resolved_db = Path(db_path).expanduser().resolve()
    digest = hashlib.sha256(str(resolved_db).encode("utf-8")).hexdigest()[:8]
    return f"全库_{digest}"


def run_duplicates(target_dir, db_path):
    """找出内容完全相同（byte 级）的重复文件，生成报告。

    只读——不删除、不移动任何文件。两种候选来源：
    - 传了 target_dir：重新扫一遍这个目录（不碰数据库，不需要先跑过
      index，随时能用），跟原来的行为完全一样。
    - 不传 target_dir：改用数据库里已经登记过的文件当候选（需要先跑过
      index），好处是能顺带发现"同一份内容分别存在两个不同目录里，各自
      都建过索引"这种跨目录重复——单独扫一个目录看不到这种情况。

    两种模式判断"是不是重复"的逻辑完全一样（只看内容 SHA256），差别只在
    候选文件从哪来，详见 duplicates.py 里 find_duplicates()/
    find_duplicates_from_index() 的说明。
    """
    import time
    from steward.duplicates import find_duplicates, find_duplicates_from_index

    if target_dir is not None:
        print(f"正在扫描 {target_dir} ...")
    else:
        print(f"没有指定目录，改从数据库（{db_path}）里已登记的文件查重复（含跨目录）...")

    t0 = time.monotonic()
    if target_dir is not None:
        groups = find_duplicates(target_dir)
    else:
        groups = find_duplicates_from_index(db_path)
    elapsed = time.monotonic() - t0

    if not groups:
        print(f"没有发现内容完全相同的重复文件。（耗时 {elapsed:.1f} 秒）")
        return

    total_wasted = sum(g[3] for g in groups)
    total_files = sum(len(g[1]) for g in groups)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"duplicates_report_{_report_tag(target_dir, db_path)}.md"

    source_desc = target_dir if target_dir is not None else f"数据库全量（{db_path}）"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 重复文件报告（{source_desc}）\n\n")
        f.write(
            f"**总计**: {len(groups)} 组重复，共 {total_files} 个文件，"
            f"可释放空间约 {_format_bytes(total_wasted)}\n\n"
        )
        for index, (digest, paths, size, wasted) in enumerate(groups, start=1):
            f.write(
                f"### {index}. {len(paths)} 份重复，单份 {_format_bytes(size)}，"
                f"浪费 {_format_bytes(wasted)}\n"
            )
            f.write(f"- **SHA256**: `{digest}`\n")
            for path in paths:
                f.write(f"  - `{path}`\n")
            f.write("\n")

    print(f"共发现 {len(groups)} 组重复，{total_files} 个文件，可释放空间约 {_format_bytes(total_wasted)}")
    print(f"耗时: {elapsed:.1f} 秒")
    print(f"✅ 报告已生成: {report_path}")


def run_duplicates_clean(target_dir, db_path):
    """交互式清理重复文件——逐组确认后，把除保留项外的文件移入系统废纸篓
    （可撤销，不是永久删除）。

    候选来源跟只读版本的 run_duplicates() 是同一套规则：传了 target_dir
    就重新扫这个目录；不传就从数据库（db_path）里已登记的文件查，能顺带
    清理跨目录的重复。

    关键行为，跟只读版本不一样：
    - 会真的改动文件系统（移入废纸篓），所以每一组都要求用户明确确认，
      不接受任何"默认全部执行"的快捷方式——delete 是这个项目第一个真正
      的"行动层"能力，宁可啰嗦也不要图快。
    - 每一次成功的删除都会追加写一条操作日志（duplicates_cleanup_log.jsonl），
      记录删的是哪个文件、保留的是哪个、什么时候删的——废纸篓本身能撤销，
      但"这次操作到底动了哪些文件"这件事不该只靠翻废纸篓才能查。
    - 这份日志记的是"某个时间点执行过这个动作"，不是"这个文件现在是不是
      还在废纸篓里"——如果用户后来自己在 Finder 里把某个文件"放回原处"
      (Put Back)还原了，日志不会跟着更新，这是预期行为，不是 bug：日志
      是操作历史，不是实时状态，就像 git commit 记录不会因为后来被 revert
      就从历史里消失一样。还原这个动作完全靠 macOS 系统自己的废纸篓机制
      (send2trash 在这台机器上实测调用的是 CoreServices 的
      FSMoveObjectToTrashSync，就是 Finder"移到废纸篓"用的同一套系统
      API，"放回原处"能力是系统原生就有的)，这里不重复实现。
    - 不碰任何不在重复分组里的文件；数据库全量模式下会读数据库拿候选
      文件列表，但从头到尾不会往数据库里写任何东西——删除文件这个动作
      跟数据库状态没有关联，DB 里的记录不会因为这次清理被更新。
    """
    import json
    from datetime import datetime

    from steward.duplicates import find_duplicates, find_duplicates_from_index, move_to_trash, pick_keeper

    if target_dir is not None:
        print(f"正在扫描 {target_dir} ...")
        groups = find_duplicates(target_dir)
    else:
        print(f"没有指定目录，改从数据库（{db_path}）里已登记的文件查重复（含跨目录）...")
        groups = find_duplicates_from_index(db_path)

    if not groups:
        print("没有发现内容完全相同的重复文件，没有可清理的。")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 这个日志是追加写（"a" 模式），不是覆盖写，所以严格说不会丢数据——但
    # 不同来源的清理记录混在同一个文件里，不方便按来源回看，跟报告文件
    # 一样按来源区分开，保持两者行为一致、可预期。
    log_path = OUTPUT_DIR / f"duplicates_cleanup_log_{_report_tag(target_dir, db_path)}.jsonl"

    total_deleted = 0
    total_freed = 0

    print(f"共发现 {len(groups)} 组重复，逐组确认。")
    print("每组输入：回车=接受默认建议 / 数字=改选要保留第几份 / s=跳过这组 / q=退出\n")

    try:
        for index, (digest, paths, size, _wasted) in enumerate(groups, start=1):
            keeper, to_delete = pick_keeper(paths)

            print(f"--- 第 {index}/{len(groups)} 组（共 {len(paths)} 份，单份 {_format_bytes(size)}）---")
            for i, path in enumerate(paths, start=1):
                try:
                    mtime_str = datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                except (FileNotFoundError, PermissionError):
                    mtime_str = "未知"
                marker = "  ← 默认建议保留" if path == keeper else ""
                print(f"  [{i}] {path}\n      修改时间: {mtime_str}{marker}")

            choice = input("请选择: ").strip().lower()

            if choice == "q":
                print("已退出，后续未处理的组不受影响。\n")
                break
            if choice == "s" or choice == "n":
                print("已跳过这一组，不做任何改动。\n")
                continue
            if choice == "":
                pass  # 接受默认的 keeper/to_delete
            elif choice.isdigit() and 1 <= int(choice) <= len(paths):
                keeper = paths[int(choice) - 1]
                to_delete = [p for p in paths if p != keeper]
            else:
                print("没识别这个输入，按跳过处理，这一组不做任何改动。\n")
                continue

            for path in to_delete:
                try:
                    move_to_trash(path)
                except Exception as e:
                    print(f"  [Warning] 移入废纸篓失败，跳过: {path}（{e}）")
                    continue

                total_deleted += 1
                total_freed += size
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "sha256": digest,
                                "kept_path": str(keeper),
                                "deleted_path": str(path),
                                "size_bytes": size,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                print(f"  已移入废纸篓: {path}")
            print()
    except KeyboardInterrupt:
        print("\n已中断，已经确认过的操作不受影响，还没处理的组不会有任何改动。")

    print("-" * 50)
    print(f"共删除 {total_deleted} 个文件，释放约 {_format_bytes(total_freed)}——都在系统废纸篓里，可以还原。")
    if total_deleted:
        print(f"操作日志: {log_path}")


def main():
    # 四个子命令都要一份一模一样的 --db 参数（路径、默认值、help 文案全部相同），
    # 之前是每个子命令各自重复写一遍。argparse 自带 parents= 机制专门解决这种
    # "多个子命令共享同一组参数"的情况：先在一个不参与解析、只当"参数模板"的
    # parser 上定义好，后面每个子命令用 parents=[db_parent] 直接继承，不用重复
    # add_argument。add_help=False 是必须的——parents 列表里的 parser 如果自己
    # 也定义了 -h/--help，会跟子命令自己的 --help 冲突报错。
    db_parent = argparse.ArgumentParser(add_help=False)
    db_parent.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite 数据库路径，默认是 {DEFAULT_DB_PATH}",
    )

    parser = argparse.ArgumentParser(description="端侧文件处理工具")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser(
        "index", help="为目录建立 document 内容索引", parents=[db_parent]
    )
    index_parser.add_argument("target_dir", help="要索引的目录，例如 ~/Documents")
    index_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "强制对所有文件重新提取/分段/向量化，忽略增量判断。默认（不传）"
            "只处理新增或内容变化过的文件。索引/提取逻辑本身变了（比如加了新"
            "的提取格式支持、新的关键词索引）之后，需要传这个才能让所有文件"
            "吃到新逻辑——增量判断只看文件内容变没变，看不出处理逻辑变没变。"
        ),
    )

    search_parser = subparsers.add_parser(
        "search", help="用自然语言搜索已建立索引的 document", parents=[db_parent]
    )
    search_parser.add_argument("query", help="搜索问题，例如 '关于 agent 学习的对话'")
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回结果数量，默认 5",
    )
    search_parser.add_argument(
        "--candidates",
        type=int,
        default=80,
        help=(
            "向量+关键词+标签粗筛阶段捞出多少个候选交给重排序模型精排，默认 80。"
            "数字越大，语料里真正相关但排名靠后的内容越不容易被粗筛漏掉，但重排序"
            "耗时跟这个数字线性增长（实测约每个候选 100~120 毫秒）。"
        ),
    )

    tag_parser = subparsers.add_parser(
        "tag", help="对已有索引的文档批量打标签（开放式 reasoning + tags）", parents=[db_parent]
    )
    tag_parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并发调用本地 SLM 的线程数，建议跟 llama-server 启动时的 -np（slot 数）匹配，默认 8",
    )
    tag_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "强制对所有文档重新打标签，忽略增量判断。默认（不传）只处理新增或"
            "内容变化过的文档，已经打过标签且内容没变的会跳过。改了 prompt/"
            "snippet 长度/模型这类打标签逻辑本身的改动之后，需要传这个才能让"
            "所有文档吃到新逻辑——增量判断只看内容变没变，看不出代码变没变。"
        ),
    )
    tag_parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("STEWARD_LLM_BASE_URL"),
        help=(
            "打标签调用的 llama-server 地址（OpenAI 兼容接口），例如 "
            "http://192.168.1.10:8080/v1。不传则用本机默认 "
            "http://127.0.0.1:8080/v1。也可以用环境变量 STEWARD_LLM_BASE_URL "
            "设一次、不用每次敲——分发给别人用时，他们机器上没跑 llama-server，"
            "必须指向一台在跑的机器。"
        ),
    )

    tags_parser = subparsers.add_parser(
        "tags", help="展示数据库中已打标签文档的标签、reasoning 及置信度", parents=[db_parent]
    )
    tags_parser.add_argument(
        "--tag",
        default=None,
        help=(
            "只看标签里包含这个关键词的文档，输出一份单独的筛选报告，不生成"
            "全量报告。不是精确匹配一个固定的标签名——底层是关键词检索（跟 "
            "search 命令关键词那一路同一套机制），查询词至少要 3 个字符才能"
            "命中；不传这个参数就是原来的行为，展示全部文档。"
        ),
    )

    duplicates_parser = subparsers.add_parser(
        "duplicates",
        help="找出内容完全相同的重复文件（默认只读；加 --clean 交互式清理）",
        description=(
            "两种候选来源，二选一，不能同时用：\n"
            "  1) 传 target_dir —— 重新扫这个目录，不碰数据库、不需要先跑过\n"
            "     index，--db 在这个模式下会被忽略，传了也没用。\n"
            "  2) 不传 target_dir —— 改从 --db 指向的数据库里已建过索引的\n"
            "     文件查重复，能顺带发现跨目录的重复（前提是那些目录之前\n"
            "     跑过 index）；--db 不传就用默认的共享数据库，通常不用管。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[db_parent],
    )
    duplicates_parser.add_argument(
        "target_dir",
        nargs="?",
        default=None,
        help="要扫描的目录，例如 ~/Downloads。不传则改走数据库模式，见上面的说明。",
    )
    duplicates_parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "交互式清理：每组重复文件逐组确认后，把除保留项外的文件移入系统"
            "废纸篓（可撤销，不是永久删除）。不加这个参数就还是原来的只读报告。"
        ),
    )

    args = parser.parse_args()

    if args.command == "index":
        run_index(args.target_dir, args.db, force=args.force)
    elif args.command == "search":
        run_search(args.query, args.db, args.top_k, args.candidates)
    elif args.command == "tag":
        run_tag(
            args.db,
            max_workers=args.workers,
            force=args.force,
            llm_base_url=args.llm_base_url,
        )
    elif args.command == "tags":
        run_tags(args.db, tag_query=args.tag)
    elif args.command == "duplicates":
        if args.clean:
            run_duplicates_clean(args.target_dir, args.db)
        else:
            run_duplicates(args.target_dir, args.db)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
