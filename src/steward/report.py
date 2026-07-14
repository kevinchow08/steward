"""汇总分类结果 + 写输出文件(逐文件明细 JSON、类别分布汇总、延迟/内存基线报告)。"""

import json


def write_results(records, output_path):
    # records 是一个 list,每个元素是一个 dict,比如:
    # {"path": "...", "basic_type": "document", "matched_by": "extension"}
    # indent=2 是为了让写出来的 JSON 文件带缩进,人眼看着方便;ensure_ascii=False 避免中文路径被转义成 \uXXXX
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def summarize(records):
    total = len(records)

    # counts 用来数每个 basic_type 出现了几次
    # dict.get(key, 0) 的意思是:如果 key 已经在 dict 里,拿它当前的值;不在的话就当 0 处理
    # 这样不用先判断 "if basic_type not in counts: counts[basic_type] = 0" 再累加
    counts = {}
    for record in records:
        basic_type = record["basic_type"]
        counts[basic_type] = counts.get(basic_type, 0) + 1

    unknown_count = counts.get("unknown", 0)

    return {
        "total_files": total,
        "by_type": counts,
        "unknown_count": unknown_count,
        # total 为 0(比如扫了个空目录)时避免除零报错
        "unknown_ratio": unknown_count / total if total else 0.0,
    }


def write_baseline_report(stats, output_path):
    # stats 是 monitor.stop() 返回的那个 dict(耗时 + 峰值内存 + 峰值 CPU),直接落盘
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
