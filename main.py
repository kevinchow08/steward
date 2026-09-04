"""入口:接收目录参数,串联 scan -> index -> tag -> report。"""

import argparse
import sys
from pathlib import Path

# main.py 在仓库根目录,包代码在 src/steward 下(src 布局)
# Python 默认不会自动把 src/ 加进模块搜索路径,得手动加,不然下面 import steward 会报 ModuleNotFoundError
# __file__ 是当前文件(main.py)自己的路径,.resolve() 转成绝对路径,.parent 拿到它所在的目录(仓库根目录)
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

RULES_PATH = BASE_DIR / "config" / "rules.yaml"
OUTPUT_DIR = BASE_DIR / "output"

# 之前这里自己又定义了一份 DEFAULT_DB_PATH（指向项目源码目录），跟
# document_index.py 里的那份是两份独立的常量、值还不一样——改 document_index.py
# 那边的默认路径完全不会影响这里，是真实的 bug，不是"两处保持一致就行"的
# 重复定义。改成直接从 document_index 引用同一个常量，只有一个真正的来源。
from steward.document_index import DEFAULT_DB_PATH  # noqa: E402


def run_week1_scan(target_dir):
    from steward import monitor, report, scan
    from steward.classifiers import rule_based

    # 创建解析器,此时它还不认识任何参数,只是个空壳
    parser = argparse.ArgumentParser(description="端侧文件类型分类(Week 1,纯规则)")

    # 登记一个参数:名字不带 "--" 前缀,所以是"位置参数"(必填,按顺序传,不用写参数名)
    # 比如 `python main.py ~/Downloads` 里的 ~/Downloads 就是传给它的值
    parser.add_argument("target_dir", help="要扫描的目录,比如 ~/Downloads")

    # 真正读 sys.argv 并按上面登记的规则解析,返回一个 Namespace 对象
    # 之后用 args.target_dir 取值;缺参数/参数名打错/多传参数,这一步会自动报错退出
    args = parser.parse_args([target_dir])

    rules = rule_based.load_rules(RULES_PATH)
    res_monitor = monitor.ResourceMonitor()

    records = []
    for file_path in scan.iter_files(args.target_dir):
        result = rule_based.classify_file(file_path, rules)
        # {"path": ..., **result}:字典解包,把 result 里的 basic_type / matched_by 两个键
        # 平铺展开合并进新字典,等价于 {"path": str(file_path), "basic_type": ..., "matched_by": ...}
        records.append({"path": str(file_path), **result})
        res_monitor.sample()

    stats = res_monitor.stop()
    summary = report.summarize(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # exist_ok=True:目录已存在也不报错
    report.write_results(records, OUTPUT_DIR / "results.json")
    # 同样是字典解包,把 stats(耗时+内存+CPU)和 summary(类别分布)两个 dict 合并成一个再写盘
    report.write_baseline_report({**stats, **summary}, OUTPUT_DIR / "baseline.json")

    print(f"共处理 {summary['total_files']} 个文件")
    print(f"类别分布: {summary['by_type']}")
    print(f"unknown 占比: {summary['unknown_ratio']:.1%}")  # :.1% 是格式化写法,把 0.333 显示成 33.3%
    print(f"耗时: {stats['elapsed_seconds']:.2f} 秒")
    print(f"峰值内存: {stats['peak_rss_mb']:.1f} MB")
    print(f"峰值 CPU: {stats['peak_cpu_percent']:.1f}%")
    print(f"结果已写入 {OUTPUT_DIR}")


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


def run_search(query, db_path, top_k):
    """加载本地模型，并在已有 document 索引中搜索，同时打印耗时与统计数据。"""

    import time
    from steward import semantic_search
    from steward.embeddings import LocalEmbedder

    total_start = time.monotonic()

    # 1. 测量本地 Embedding 模型加载耗时
    print("正在加载本地 embedding 模型...")
    t_model_start = time.monotonic()
    embedder = LocalEmbedder()
    model_load_seconds = time.monotonic() - t_model_start

    # 2. 执行语义搜索，接收结果和详细耗时/计数
    results, stats = semantic_search.search_documents(
        query,
        embedder,
        db_path=db_path,
        top_k=top_k,
    )

    total_seconds = time.monotonic() - total_start

    if not results:
        print("没有找到结果。")
    else:
        for index, result in enumerate(results, start=1):
            # 这个 score 是向量检索+关键词检索融合排序（RRF）之后的分数，不是
            # 相似度百分比，数值本身很小、没有直接意义，只用来决定排在第几，
            # 不要按"匹配度多少%"去解读。
            print(f"{index}. 融合排序分数={result.score:.4f}")
            print(f"   文件: {result.path}")
            print(f"   chunk: {result.chunk_index}")
            print(f"   片段: {result.text}")

    # 3. 打印性能与耗时监控信息
    print("-" * 50)
    print("【性能与耗时统计】")
    print(f"最终命中文档: {stats['document_count']} 个 | 比较片段(chunks): {stats['chunk_count']} 个")
    print(f"向量检索命中: {stats['dense_hit_count']} 个文档 | 关键词检索命中: {stats['sparse_hit_count']} 个文档")
    print(f"模型加载耗时: {model_load_seconds:.3f} 秒")
    print(f"Query 向量化: {stats['query_embed_seconds']:.3f} 秒")
    print(f"向量检索耗时: {stats['vector_search_seconds']:.3f} 秒")
    print(f"关键词检索耗时: {stats['keyword_search_seconds']:.3f} 秒")
    print(f"搜索总耗时:   {total_seconds:.3f} 秒")


def run_tag(db_path, max_workers=8, force=False):
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
    """

    import time
    from steward.document_index import DocumentIndex
    from steward.tagging import run_tagging_pipeline

    print(f"🚀 启动打标签引擎（开放式 reasoning + tags{'，force 模式：全部重新打标签' if force else ''}）...")
    start_time = time.monotonic()

    with DocumentIndex(db_path) as index:
        stats = run_tagging_pipeline(
            index=index,
            max_workers=max_workers,
            force=force,
        )

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
    print(f"steward 自身进程 峰值内存: {stats['steward_peak_rss_mb']:.1f} MB | 峰值 CPU: {stats['steward_peak_cpu_percent']:.1f}%")
    if stats["llama_server_peak_rss_mb"] is not None:
        print(f"llama-server 进程 峰值内存: {stats['llama_server_peak_rss_mb']:.1f} MB | "
              f"峰值 CPU: {stats['llama_server_peak_cpu_percent']:.1f}%")
    else:
        print("llama-server 进程: 没找到，跳过这一路监控（GPU 算力占用暂不测，见 monitor.py 的注释）")
    print(f"数据库持久化: {db_path}")



def run_tags(db_path):
    """展示 SQLite 中已打标签文档的标签、reasoning 及置信度，并输出到文件。"""

    from steward.document_index import DocumentIndex

    with DocumentIndex(db_path) as index:
        records = list(index.iter_tagging_results())

    if not records:
        print("当前没有任何已打标签的文档。请先运行: python main.py tag")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "tags_report.md"

    # 三种状态分开统计，报告里也分开展示——"深度打标签"和"基础类型识别"的可信程度
    # 不一样，混在一起看容易把后者的粗糙标签误当成前者那种经过语义判断的结果。
    status_icon = {"tagged": "✅", "basic": "🔹", "untagged": "⚠️"}
    status_label = {"tagged": "深度打标签", "basic": "基础类型识别", "untagged": "未打标签"}
    counts = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print(f"共查找到 {len(records)} 份文档。正在生成报告...")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 文档打标签报告\n\n")
        f.write(f"**总计**: {len(records)} 份文档")
        f.write("（" + "，".join(f"{status_label.get(s, s)} {n} 份" for s, n in counts.items()) + "）\n\n")
        for index, r in enumerate(records, start=1):
            tags_str = ", ".join(r["tags"]) if r["tags"] else "(无标签)"
            icon = status_icon.get(r["status"], "❔")
            f.write(f"### {index}. {icon} {status_label.get(r['status'], r['status'])}\n")
            f.write(f"- **标签**: {tags_str}\n")
            f.write(f"- **置信度**: {r['confidence']:.2f}\n")
            f.write(f"- **文件**: `{r['path']}`\n")
            f.write(f"- **依据**: {r['reasoning']}\n\n")

    print(f"✅ 报告已生成: {report_path}")


def main():
    # 保留 Week 1 的旧用法：python main.py ~/Downloads
    # 新功能使用子命令：python main.py index ~/Documents
    if len(sys.argv) > 1 and sys.argv[1] not in {"index", "search", "tag", "tags", "-h", "--help"}:
        run_week1_scan(sys.argv[1])
        return

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

    tags_parser = subparsers.add_parser(
        "tags", help="展示数据库中已打标签文档的标签、reasoning 及置信度", parents=[db_parent]
    )

    args = parser.parse_args()

    if args.command == "index":
        run_index(args.target_dir, args.db, force=args.force)
    elif args.command == "search":
        run_search(args.query, args.db, args.top_k)
    elif args.command == "tag":
        run_tag(
            args.db,
            max_workers=args.workers,
            force=args.force,
        )
    elif args.command == "tags":
        run_tags(args.db)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
