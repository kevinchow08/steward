"""入口:接收目录参数,串联 scan -> classify -> monitor -> report。"""

import argparse
import sys
from pathlib import Path

# main.py 在仓库根目录,包代码在 src/steward 下(src 布局)
# Python 默认不会自动把 src/ 加进模块搜索路径,得手动加,不然下面 import steward 会报 ModuleNotFoundError
# __file__ 是当前文件(main.py)自己的路径,.resolve() 转成绝对路径,.parent 拿到它所在的目录(仓库根目录)
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from steward import monitor, report, scan  # noqa: E402(必须在 sys.path 改好之后才能 import,顺序不能换)
from steward.classifiers import rule_based  # noqa: E402

RULES_PATH = BASE_DIR / "config" / "rules.yaml"
OUTPUT_DIR = BASE_DIR / "output"


def main():
    # 创建解析器,此时它还不认识任何参数,只是个空壳
    parser = argparse.ArgumentParser(description="端侧文件类型分类(Week 1,纯规则)")

    # 登记一个参数:名字不带 "--" 前缀,所以是"位置参数"(必填,按顺序传,不用写参数名)
    # 比如 `python main.py ~/Downloads` 里的 ~/Downloads 就是传给它的值
    parser.add_argument("target_dir", help="要扫描的目录,比如 ~/Downloads")

    # 真正读 sys.argv 并按上面登记的规则解析,返回一个 Namespace 对象
    # 之后用 args.target_dir 取值;缺参数/参数名打错/多传参数,这一步会自动报错退出
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
