"""入口:接收目录参数,串联 scan -> classify -> monitor -> report。"""

import argparse
import sys
from pathlib import Path

# main.py 在仓库根目录,包代码在 src/steward 下(src 布局)
# Python 默认不会自动把 src/ 加进模块搜索路径,得手动加,不然下面 import steward 会报 ModuleNotFoundError
# __file__ 是当前文件(main.py)自己的路径,.resolve() 转成绝对路径,.parent 拿到它所在的目录(仓库根目录)
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

RULES_PATH = BASE_DIR / "config" / "rules.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_DB_PATH = PROJECT_ROOT / "steward.db"


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


def run_index(target_dir, db_path):
    """加载本地模型并为目录建立 document 内容索引。"""

    from steward import indexing
    from steward.embeddings import LocalEmbedder

    print("正在加载本地 embedding 模型，首次运行可能需要下载模型文件...")
    embedder = LocalEmbedder()
    stats = indexing.build_index(target_dir, embedder, db_path=db_path)

    print(f"扫描文件: {stats['scanned_files']}")
    print(f"已建立索引: {stats['indexed_files']}")
    print(f"成功提取: {stats['success_files']}")
    print(f"无文字内容: {stats['no_text_files']}")
    print(f"提取失败: {stats['error_files']}")
    print(f"暂不支持: {stats['unsupported_files']}")
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
            print(f"{index}. score={result.score:.4f}")
            print(f"   文件: {result.path}")
            print(f"   chunk: {result.chunk_index}")
            print(f"   片段: {result.text}")

    # 3. 打印性能与耗时监控信息
    print("-" * 50)
    print("【性能与耗时统计】")
    print(f"扫描比较文件: {stats['document_count']} 个 | 比较片段(chunks): {stats['chunk_count']} 个")
    print(f"模型加载耗时: {model_load_seconds:.3f} 秒")
    print(f"Query 向量化: {stats['query_embed_seconds']:.3f} 秒")
    print(f"向量对比计算: {stats['vector_search_seconds']:.3f} 秒")
    print(f"搜索总耗时:   {total_seconds:.3f} 秒")


def run_classify(db_path):
    """为已有索引文本的文档批量执行 100% 动态无监督分类与多标签打标，并将结果持久化。"""

    import time
    from steward.document_index import DocumentIndex
    from steward.semantic_classifier import run_dynamic_classification_pipeline

    print("🚀 启动 Phase 3 端到端 100% 动态无规则分类与多标签打标引擎...")
    start_time = time.monotonic()

    with DocumentIndex(db_path) as index:
        stats = run_dynamic_classification_pipeline(index=index)

    elapsed = time.monotonic() - start_time

    print("-" * 50)
    print("【100% 动态无监督分类与打标签统计】")
    print(f"分析文档总数: {stats['total_documents']} 份")
    print(f"聚类发现簇数: {stats['clusters']} 个主题簇 (离群孤立文件: {stats['outliers']} 份)")
    print(f"成功打标文件: {stats['tagged_documents']} 份")
    print(f"全管线总耗时: {elapsed:.3f} 秒")
    print(f"数据库持久化: {db_path}")



def run_tags(db_path):
    """展示 SQLite 中已分类文档的主分类、置信度及关联标签列表，并输出到文件。"""

    from steward.document_index import DocumentIndex

    with DocumentIndex(db_path) as index:
        records = list(index.iter_classifications())

    if not records:
        print("当前没有任何已分类打标的文档。请先运行: python main.py classify")
        return


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "tags_report.md"

    print(f"共查找到 {len(records)} 份已分类文档。正在生成报告...")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 文档分类与标签报告\n\n")
        f.write(f"**总计**: {len(records)} 份文档\n\n")
        for index, r in enumerate(records, start=1):
            tags_str = ", ".join(r["tags"]) if r["tags"] else "(无标签)"
            status_flag = "✅" if r["status"] == "classified" else "⚠️"
            f.write(f"### {index}. {status_flag} 分类: [{r['category']}]\n")
            f.write(f"- **置信度**: {r['confidence']:.2f}\n")
            f.write(f"- **文件**: `{r['path']}`\n")
            f.write(f"- **标签**: {tags_str}\n")
            f.write(f"- **依据**: {r['reasoning']}\n\n")

    print(f"✅ 报告已生成: {report_path}")


def main():
    # 保留 Week 1 的旧用法：python main.py ~/Downloads
    # 新功能使用子命令：python main.py index ~/Documents
    if len(sys.argv) > 1 and sys.argv[1] not in {"index", "search", "classify", "tags", "-h", "--help"}:
        run_week1_scan(sys.argv[1])
        return

    parser = argparse.ArgumentParser(description="端侧文件处理工具")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="为目录建立 document 内容索引")
    index_parser.add_argument("target_dir", help="要索引的目录，例如 ~/Documents")
    index_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite 数据库路径，默认是 {DEFAULT_DB_PATH}",
    )

    search_parser = subparsers.add_parser("search", help="用自然语言搜索已建立索引的 document")
    search_parser.add_argument("query", help="搜索问题，例如 '关于 agent 学习的对话'")
    search_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite 数据库路径，默认是 {DEFAULT_DB_PATH}",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回结果数量，默认 5",
    )

    classify_parser = subparsers.add_parser("classify", help="对已有索引的 document 批量执行语义分类与打标签")
    classify_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite 数据库路径，默认是 {DEFAULT_DB_PATH}",
    )

    tags_parser = subparsers.add_parser("tags", help="展示数据库中已分类文档的主分类、置信度及标签列表")
    tags_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite 数据库路径，默认是 {DEFAULT_DB_PATH}",
    )

    args = parser.parse_args()

    if args.command == "index":
        run_index(args.target_dir, args.db)
    elif args.command == "search":
        run_search(args.query, args.db, args.top_k)
    elif args.command == "classify":
        run_classify(args.db)
    elif args.command == "tags":
        run_tags(args.db)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
