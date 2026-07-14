"""汇总分类结果 + 写输出文件(逐文件明细 JSON、类别分布汇总、延迟/内存基线报告)。"""


def write_results(records, output_path):
    # TODO: 把逐文件分类记录写成 JSON,字段带阶段前缀(如 basic_type),方便 Week 2+ 复用
    raise NotImplementedError


def summarize(records):
    # TODO: 统计各 basic_type 的数量/占比,以及 unknown 的比例
    raise NotImplementedError


def write_baseline_report(stats, output_path):
    # TODO: 写延迟 + 内存占用的基线报告
    raise NotImplementedError
