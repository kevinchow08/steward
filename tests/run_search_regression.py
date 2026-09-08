"""语义搜索回归测试——不是单元测试框架意义上的"测试"（这个项目没有引入
pytest 之类的测试库，保持"能不加依赖就不加"的一贯原则），是一份可以反复
重跑、自动判定通过/失败的脚本，取代"每次改动完手动敲几个查询词、肉眼看
结果"这种没法留存、没法对比的验证方式。

案例定义在 search_regression_cases.json 里，跟这份脚本分开放——加/改
案例不需要碰代码，方便随时扩充。

用法：
    python tests/run_search_regression.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steward import semantic_search
from steward.embeddings import LocalEmbedder
from steward.reranker import LocalReranker

CASES_PATH = Path(__file__).resolve().parent / "search_regression_cases.json"


def check_case(case, results):
    """按案例定义的期待，判断这次搜索结果是通过还是失败。

    返回 (passed, detail)：passed 是 True/False/None（None 代表这条案例
    本来就不做严格判定，只是记录，见 "note_only" 那种案例类型）。
    """
    expect = case["expect"]

    if expect == "empty":
        if not results:
            return True, "正确返回空结果"
        return False, f"期待空结果，实际返回了 {len(results)} 个（第一名: {results[0].path}）"

    if expect == "note_only":
        top = results[0].path if results else "(空)"
        return None, f"仅记录，不判定：第一名 = {top}"

    if expect == "match":
        if not results:
            return False, "期待有结果，实际是空"

        top_path = results[0].path
        ok_substrings = case.get("top_result_should_contain")
        if ok_substrings and not any(s in top_path for s in ok_substrings):
            return False, f"第一名 {top_path!r} 不包含任何期待的关键词 {ok_substrings}"

        bad_substrings = case.get("should_not_appear_in_top", [])
        for result in results:
            for bad in bad_substrings:
                if bad in result.path:
                    return False, f"结果里混入了已知的错误案例: {result.path!r}（含关键词 {bad!r}）"

        return True, f"第一名: {top_path}"

    raise ValueError(f"不认识的 expect 类型: {expect!r}")


def main():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    print("正在加载本地模型（embedding + 重排序）...")
    embedder = LocalEmbedder()
    reranker = LocalReranker()

    passed_count = 0
    failed_count = 0
    noted_count = 0

    for case in cases:
        query = case["query"]
        results, stats = semantic_search.search_documents(query, embedder, reranker)

        passed, detail = check_case(case, results)

        gap = stats["top1_top2_gap"]
        gap_str = "inf" if gap == float("inf") else (f"{gap:.3f}" if gap is not None else "N/A")

        if passed is None:
            icon = "📝"
            noted_count += 1
        elif passed:
            icon = "✅"
            passed_count += 1
        else:
            icon = "❌"
            failed_count += 1

        print(f"{icon} {query!r}")
        print(f"    {detail}")
        print(f"    top1 分数={stats['top1_score']} | top1-top2 分差={gap_str} | 通过重排序={stats['document_count']} 个")
        print()

    total_judged = passed_count + failed_count
    print("-" * 50)
    print(f"通过 {passed_count}/{total_judged}（另有 {noted_count} 条仅记录不判定）")
    if failed_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
