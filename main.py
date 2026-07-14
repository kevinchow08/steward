"""入口:接收目录参数,串联 scan -> classify -> monitor -> report。"""

import argparse


def main():
    # 创建解析器,此时它还不认识任何参数,只是个空壳
    parser = argparse.ArgumentParser(description="端侧文件类型分类(Week 1,纯规则)")

    # 登记一个参数:名字不带 "--" 前缀,所以是"位置参数"(必填,按顺序传,不用写参数名)
    # 比如 `python main.py ~/Downloads` 里的 ~/Downloads 就是传给它的值
    parser.add_argument("target_dir", help="要扫描的目录,比如 ~/Downloads")

    # 真正读 sys.argv 并按上面登记的规则解析,返回一个 Namespace 对象
    # 之后用 args.target_dir 取值;缺参数/参数名打错/多传参数,这一步会自动报错退出
    args = parser.parse_args()

    # TODO: 串联 scan.iter_files -> classifiers.rule_based.classify_file
    #       -> monitor.ResourceMonitor -> report.write_results / write_baseline_report
    raise NotImplementedError


if __name__ == "__main__":
    main()
