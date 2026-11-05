"""检索评测脚本——用固定的评测集（search_eval_cases.json）给当前检索打分，
用来回答"一次改动之后，检索是变好了还是变坏了"。

跟以前的回归脚本（只看第 1 名、命中任意一个关键词就算过）不同，这里：
- 每条用例标注的是"全部"正确答案的文件路径，不只是第 1 名该是谁；
- 算的是排序指标（Hit@1 / Hit@5 / MRR / Recall@10），正确文件从第 1 名掉到
  第 4 名、或者被门槛过滤掉，分数都会如实变化，不会被"宽松判定"盖住；
- 按查询类型分组统计，能看出问题集中在哪一类查询上。

指标解释（只对有正确答案的用例算，负例单独算）：
- Hit@1：第 1 名是不是正确文件。
- Hit@5：前 5 名里有没有至少一个正确文件。
- MRR（Mean Reciprocal Rank，平均倒数排名）：第一个正确文件排第 r 名就得 1/r 分，
  第 1 名得 1、第 2 名得 0.5、第 5 名得 0.2，前 10 名都没有得 0，再对所有用例取
  平均。比 Hit@k 更细，能看出"排名往前挪了几位"这种变化。
- Recall@10：前 10 名里找回了多少个正确文件，除以 min(正确文件总数, 10)——
  正确文件有 40 个时，前 10 名全对就算满分，不会因为只能展示 10 个而永远拿不到满分。
- 负例误报率：语料里没有的话题，本应返回空，实际却返回了结果的比例。

⚠️ 必须用 --db 指向一份数据库快照，不要直接用真实库——真实库会被 launchd
定时刷新，语料一变，前后两次分数就没法比。

用法：
    python tests/run_search_eval.py --db <快照路径>                 # 默认只跑调参集
    python tests/run_search_eval.py --db <快照路径> --split all
    python tests/run_search_eval.py --db <快照路径> --out run_b.json --compare run_a.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steward import semantic_search
from steward.document_index import DocumentIndex

CASES_PATH = Path(__file__).resolve().parent / "search_eval_cases.json"
HOME = str(Path.home()) + "/"

# 评测只看前 10 名——Hit@5 / MRR / Recall@10 都在这个范围内算。
TOP_K = 10


def load_present_paths(db_path):
    """数据库里当前存在的全部文件路径（相对主目录）。用来展开"文件夹"条目，
    以及检查标注的路径是不是还存在（文件被移走/改名后，标注就过期了）。"""
    with DocumentIndex(db_path) as index:
        rows = index.connection.execute("SELECT path FROM documents WHERE is_present = 1").fetchall()
    return {row[0].replace(HOME, "", 1) for row in rows}


def resolve_labels(case, present_paths):
    """把用例里的 relevant / acceptable 展开成具体文件集合。

    以 / 结尾的条目代表"这个文件夹下的全部文件"。同一个文件同时被两边覆盖时，
    按"写得越具体越优先"判定：
        明确写出的 relevant > 明确写出的 acceptable > 文件夹 relevant > 文件夹 acceptable
    例如 D02 把整个"2025二季度发票/"文件夹标成 relevant，但又单独把其中的
    经费申请表标成 acceptable——那张表最终算 acceptable。
    """
    def split_entries(entries):
        files = {e for e in entries if not e.endswith("/")}
        folders = [e for e in entries if e.endswith("/")]
        return files, folders

    rel_files, rel_folders = split_entries(case["relevant"])
    acc_files, acc_folders = split_entries(case["acceptable"])

    relevant, acceptable, missing = set(), set(), []
    for f in rel_files | acc_files:
        if f not in present_paths:
            missing.append(f)

    for path in present_paths:
        if path in rel_files:
            relevant.add(path)
        elif path in acc_files:
            acceptable.add(path)
        elif any(path.startswith(folder) for folder in rel_folders):
            relevant.add(path)
        elif any(path.startswith(folder) for folder in acc_folders):
            acceptable.add(path)

    return relevant, acceptable, missing


def score_case(ranked_paths, relevant, acceptable):
    """给一条有正确答案的用例打分。acceptable 的文件先从结果里剔除、不占名次——
    出现了不算错，也不算对。"""
    judged = [p for p in ranked_paths if p not in acceptable][:TOP_K]
    first_hit = next((i + 1 for i, p in enumerate(judged) if p in relevant), None)
    found = sum(1 for p in judged if p in relevant)
    return {
        "hit1": 1.0 if first_hit == 1 else 0.0,
        "hit5": 1.0 if first_hit is not None and first_hit <= 5 else 0.0,
        "mrr": 1.0 / first_hit if first_hit else 0.0,
        "recall10": found / min(len(relevant), TOP_K),
        "first_hit_rank": first_hit,
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def print_summary(records):
    """按查询类型分组打印指标，最后一行是全部正例的总体平均。"""
    positives = [r for r in records if r["type"] != "negative"]
    negatives = [r for r in records if r["type"] == "negative"]

    by_type = defaultdict(list)
    for r in positives:
        by_type[r["type"]].append(r)

    print(f"{'类型':<14}{'条数':>4}{'Hit@1':>8}{'Hit@5':>8}{'MRR':>8}{'R@10':>8}")
    rows = list(by_type.items()) + [("【正例总体】", positives)]

    # 用 + 拼接后：
    # rows = [
    #     ("known_item",   [K01的结果, K02的结果]),
    #     ("topic",        [T01的结果]),
    #     ("【正例总体】",  [K01的结果, K02的结果, T01的结果]),
    # ]

    for name, group in rows:
        print(f"{name:<14}{len(group):>4}"
              f"{mean([r['metrics']['hit1'] for r in group]):>8.2f}"
              f"{mean([r['metrics']['hit5'] for r in group]):>8.2f}"
              f"{mean([r['metrics']['mrr'] for r in group]):>8.2f}"
              f"{mean([r['metrics']['recall10'] for r in group]):>8.2f}")
    if negatives:
        false_alarms = sum(1 for r in negatives if r["result_count"] > 0)
        print(f"负例误报：{false_alarms}/{len(negatives)}")


def print_compare(records, previous_path):
    """跟上一次的运行结果逐条对比，只打印第一个正确文件名次有变化、或者负例
    结论有变化的用例——看清楚"这次改动具体修好了谁、搞坏了谁"。"""
    previous = {r["id"]: r for r in json.loads(Path(previous_path).read_text(encoding="utf-8"))["records"]}
    print(f"\n【与 {previous_path} 对比，只列出有变化的用例】")
    changed = 0
    for r in records:
        old = previous.get(r["id"])
        if old is None:
            continue
        if r["type"] == "negative":
            before, after = old["result_count"], r["result_count"]
            label = "返回结果数"
        else:
            before, after = old["metrics"]["first_hit_rank"], r["metrics"]["first_hit_rank"]
            label = "第一个正确文件的名次"
        if before != after:
            changed += 1
            print(f"  {r['id']} {r['query']!r}: {label} {before} -> {after}")
    if not changed:
        print("  （没有变化）")


def main():
    parser = argparse.ArgumentParser(description="检索评测")
    parser.add_argument("--db", required=True, help="数据库快照路径（不要用会被自动刷新的真实库）")
    parser.add_argument("--split", choices=["tune", "holdout", "all"], default="tune")
    parser.add_argument("--out", help="把这次的逐条结果存成 JSON，供以后 --compare 用")
    parser.add_argument("--compare", help="上一次 --out 存下的 JSON，逐条对比名次变化")
    parser.add_argument("--candidates", type=int, default=semantic_search.DEFAULT_CANDIDATE_POOL_SIZE)
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    if args.split != "all":
        cases = [c for c in cases if c["split"] == args.split]

    present_paths = load_present_paths(args.db)

    # 延迟导入：模型库加载慢，参数写错时能更快报错退出。
    from steward.embeddings import LocalEmbedder
    from steward.reranker import LocalReranker
    print("正在加载本地模型（embedding + 重排序）...")
    embedder = LocalEmbedder()
    reranker = LocalReranker()

    records = []
    for case in cases:
        relevant, acceptable, missing = resolve_labels(case, present_paths)
        if missing:
            print(f"⚠️  {case['id']} 有 {len(missing)} 个标注路径在数据库里找不到（标注过期？）：{missing[:3]}")

        t0 = time.monotonic()
        results, stats = semantic_search.search_documents(
            case["query"], embedder, reranker,
            db_path=args.db, top_k=TOP_K, candidate_pool_size=args.candidates,
        )
        seconds = time.monotonic() - t0
        ranked_paths = [r.path.replace(HOME, "", 1) for r in results]

        record = {
            "id": case["id"], "query": case["query"], "type": case["type"], "split": case["split"],
            "result_count": len(results), "seconds": round(seconds, 2),
            "top": [{"path": p, "score": round(r.score, 3),
                     "label": "relevant" if p in relevant else "acceptable" if p in acceptable else "-"}
                    for p, r in zip(ranked_paths, results)],
        }
        if case["type"] == "negative":
            mark = "✅" if not results else f"❌ 返回了 {len(results)} 个"
        else:
            if not relevant:
                print(f"⚠️  {case['id']} 展开后没有任何正确文件，跳过")
                continue
            record["metrics"] = score_case(ranked_paths, relevant, acceptable)
            rank = record["metrics"]["first_hit_rank"]
            mark = f"第一个正确文件排第 {rank} 名" if rank else "❌ 前 10 名没有正确文件"
        records.append(record)
        print(f"{case['id']} {case['query']!r}: {mark}（{seconds:.1f}s）")

    print("-" * 50)
    print(f"split={args.split}，共 {len(records)} 条")
    print_summary(records)

    if args.compare:
        print_compare(records, args.compare)

    if args.out:
        Path(args.out).write_text(
            json.dumps({"split": args.split, "db": args.db, "records": records}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"\n逐条结果已保存到 {args.out}")


if __name__ == "__main__":
    main()
