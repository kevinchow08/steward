"""数据库管理目录的自动增量刷新，串联"对每个已管理目录跑增量 index" +
"跑一次全库增量 tag"（tag 本来就不分目录，见 cli.py 的 run_tag）。

设计背景见 docs/product_direction_and_roadmap.md 里"数据库管理目录的自动
增量刷新"那段讨论：方案是"launchd 触发一次性命令"，不是常驻进程——不用的
时候零资源占用，需要跑的时候由操作系统负责叫醒。具体触发方式用 launchd
的 WatchPaths（数据库管理的目录有变化就触发），不是简单的定时轮询，配合
下面的节流机制避免用户高强度改文件时被连续拉起来跑好几次。

三个真实边界情况，对应下面三处设计：
1. 上次没跑完，这次又被触发了（重叠保护）——PID 锁文件 + psutil 判活。
2. llama-server 没启动，打标签不能拖垮整个任务——跑 tag 前先探测 /health。
3. 失败要写日志，不能悄悄丢失——launchd 触发的任务没有终端可看，refresh.log
   是唯一能事后查的地方，不管这次是正常完成、被节流跳过、还是被锁挡住，
   都要留一条记录。
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psutil

from steward.document_index import DocumentIndex
from steward.paths import (
    DEFAULT_DB_PATH,
    REFRESH_LAST_RUN_PATH,
    REFRESH_LOCK_PATH,
    REFRESH_LOG_PATH,
)

# WatchPaths 可能在用户高强度编辑文件时被连续触发好几次，这个窗口内的重复
# 触发直接跳过，不重新走一遍"加载模型+扫描"的完整流程。这个数字不是精确
# 算出来的，是"比一次正常 refresh 的耗时宽松几倍"的经验值，之后发现不合适
# 再调（调大/调小都只影响这一个常量，不涉及其它逻辑）。
THROTTLE_SECONDS = 30 * 60

LAUNCHD_LABEL = "com.steward.refresh"
LAUNCHD_PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _log(message):
    """追加写一行日志，带时间戳。REFRESH_LOG_PATH 是这次运行结果唯一能
    事后查的地方（launchd 触发的任务没有终端可看），所以这个函数在
    run_refresh() 的每一个分支（正常完成/节流跳过/锁挡住/目录不可达/
    llama-server 不在线）都要被调用一次，不能有任何"安静地结束"的路径。
    """
    REFRESH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REFRESH_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{_utc_now_iso()}] {message}\n")


def _check_throttle():
    """距离上次真正开始跑还不到 THROTTLE_SECONDS，返回 True（应该跳过）。"""
    if not REFRESH_LAST_RUN_PATH.exists():
        return False
    try:
        last_started = datetime.fromisoformat(REFRESH_LAST_RUN_PATH.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        # 状态文件读不出来/格式坏了，保守起见不拦，正常跑一次顺便把它覆盖成正确格式。
        return False
    elapsed = (datetime.now(timezone.utc) - last_started).total_seconds()
    return elapsed < THROTTLE_SECONDS


def _acquire_lock():
    """PID 锁：已经有一次真的在跑就返回 False（应该跳过）；上次异常崩溃
    留下的死锁（记录的 PID 已经不存在了）自动清理掉，返回 True 继续跑。
    """
    REFRESH_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REFRESH_LOCK_PATH.exists():
        try:
            old_pid = int(REFRESH_LOCK_PATH.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            old_pid = None
        if old_pid is not None and psutil.pid_exists(old_pid):
            return False
        if old_pid is None:
            _log("发现锁文件内容异常（读不出有效的 PID），已忽略并清理，继续本次运行。")
        else:
            _log(f"发现上次异常退出留下的锁文件（记录的 PID {old_pid} 已不存在），已清理，继续本次运行。")
    REFRESH_LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _release_lock():
    # unlink() 删除的是整个文件（不是清空内容留一个空文件），删完之后
    # REFRESH_LOCK_PATH.exists() 会变成 False，下次 _acquire_lock() 就不会
    # 走"锁文件存在，要不要判断死锁"这条分支，直接写自己的 PID 拿到锁。
    try:
        REFRESH_LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def _llama_server_healthy(llm_base_url):
    """探测 llama-server 是否在线，5 秒超时。不在线就让调用方直接跳过打
    标签这一步，不要让每一份待打标签的文档都空等两次失败重试——tagging.py
    里每份文档本身就有 2 次重试，llama-server 没启动的话，这个代价会被
    放大到全部待处理文档，纯粹浪费时间。

    超时原本设的是 2 秒，真实撞过一次误判：refresh 触发的那一刻，
    llama-server 正在处理一个真实的推理任务（单次请求总耗时超过 1 秒，
    见真实日志），2 秒的窗口偶尔会被这种"服务器在但正忙"的情况顶到，
    被误判成"不在线"，代价是那一轮保守跳过了打标签（不是崩溃/丢数据，
    只是那一轮没打成）。调宽到 5 秒降低这种误判概率——服务器真的没启动
    时是连接直接被拒绝，不会等到超时，所以调宽这个值不会让"真的不在线"
    的判断变慢，只影响"在线但正忙"这种边界情况的容忍度。
    """
    base = llm_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    health_url = base.rstrip("/") + "/health"
    try:
        resp = httpx.get(health_url, timeout=5.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def run_refresh(db_path=DEFAULT_DB_PATH, llm_base_url=None):
    """跑一次增量刷新。返回一份统计字典，供 CLI 打印；不管结果如何，都会
    往 REFRESH_LOG_PATH 追加一条摘要。
    """
    from steward.tagging import DEFAULT_LLM_BASE_URL

    llm_base_url = llm_base_url or DEFAULT_LLM_BASE_URL

    if _check_throttle():
        _log("触发被节流：距离上次运行不到设定的时间窗口，本次跳过。")
        return {"skipped": "throttled"}

    if not _acquire_lock():
        _log("触发被重叠保护挡住：上一次 refresh 还在跑，本次跳过。")
        return {"skipped": "already_running"}

    REFRESH_LAST_RUN_PATH.write_text(_utc_now_iso(), encoding="utf-8")

    started_at = time.monotonic()
    result = {
        "processed_dirs": [],
        "unreachable_dirs": [],
        "tag_stats": None,
        "tag_skipped_reason": None,
    }

    try:
        with DocumentIndex(db_path) as index:
            target_dirs = index.list_target_dirs()

        if not target_dirs:
            _log("数据库里还没有任何被管理的目录（从没跑过 index），本次没有可刷新的内容。")
        else:
            from steward import indexing
            from steward.embeddings import LocalEmbedder

            embedder = LocalEmbedder()
            for target_dir in target_dirs:
                try:
                    stats = indexing.build_index(target_dir, embedder, db_path=db_path, force=False)
                    result["processed_dirs"].append((target_dir, stats))
                    _log(
                        f"index 完成: {target_dir}（扫描 {stats['scanned_files']}，"
                        f"跳过未变化 {stats['skipped_unchanged_files']}，"
                        f"标记消失 {stats['removed_files']}）"
                    )
                except FileNotFoundError as e:
                    # 最常见的原因：外置盘这次没连接。不能让一个目录不可达
                    # 中断其它目录的刷新。
                    result["unreachable_dirs"].append(target_dir)
                    _log(f"[Warning] 目录不可达，跳过（可能是外置盘没连接）: {target_dir}（{e}）")
                except Exception as e:
                    # 无人值守的后台任务，跟 tagging.py 逐文档 try/except 的
                    # 出发点一样：一个目录出意外不该拖垮其它目录，先记下来，
                    # 继续处理剩下的目录。
                    result["unreachable_dirs"].append(target_dir)
                    _log(f"[Warning] 处理该目录时出现意外错误，跳过，不影响其它目录: {target_dir}（{e}）")

        if _llama_server_healthy(llm_base_url):
            try:
                from steward.tagging import run_tagging_pipeline

                with DocumentIndex(db_path) as index:
                    tag_stats = run_tagging_pipeline(index=index, llm_base_url=llm_base_url)
                result["tag_stats"] = tag_stats
                _log(
                    f"tag 完成: 深度打标签 {tag_stats['tagged_count']} 份，"
                    f"内容没变跳过 {tag_stats['skipped_up_to_date_count']} 份"
                )
            except Exception as e:
                result["tag_skipped_reason"] = f"打标签过程中出现意外错误: {e}"
                _log(f"[Warning] 打标签过程中出现意外错误，本次 index 部分的结果不受影响: {e}")
        else:
            result["tag_skipped_reason"] = "llama-server 不在线"
            _log(f"llama-server（{llm_base_url}）不在线，本次跳过打标签，只完成了 index 部分。")

        elapsed = time.monotonic() - started_at
        result["elapsed_seconds"] = elapsed
        _log(
            f"本次 refresh 完成，耗时 {elapsed:.1f} 秒，"
            f"处理 {len(result['processed_dirs'])} 个目录，"
            f"{len(result['unreachable_dirs'])} 个目录不可达/出错。"
        )
    finally:
        _release_lock()

    return result


def _xml_escape(text):
    """plist 是 XML，文件路径里理论上可能出现 & < > 这几个 XML 特殊字符
    （比如文件夹名字带 &），不转义会生成一份解析不了的 plist。"""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_plist_content(steward_executable, target_dirs, stdout_path, stderr_path):
    watch_paths_xml = "\n".join(f"        <string>{_xml_escape(d)}</string>" for d in target_dirs)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{_xml_escape(steward_executable)}</string>
        <string>refresh</string>
    </array>

    <key>WatchPaths</key>
    <array>
{watch_paths_xml}
    </array>

    <key>StandardOutPath</key>
    <string>{_xml_escape(stdout_path)}</string>
    <key>StandardErrorPath</key>
    <string>{_xml_escape(stderr_path)}</string>
</dict>
</plist>
"""


def _find_steward_executable():
    """定位当前环境里真正安装出来的 steward 入口脚本，返回绝对路径字符串。

    真实踩过的坑：最初这里直接用 sys.argv[0]，理由是"这次调用本来就是
    通过装好的 steward 命令触发的"——这个假设在开发时用
    `python main.py schedule-setup` 触发（仓库根目录那个薄壳）时不成立，
    sys.argv[0] 会是 main.py 自己的路径，不是真正的 steward 入口脚本。
    main.py 本身没有可执行权限、也没有 shebang，结果是 plist 里写进了一个
    launchd 根本执行不了的路径——真实后果：macOS "登录项与扩展"面板里能
    看到一个来路不明的"main.py"，而且这个任务一旦被触发就会执行失败。

    改用 sys.executable——它是当前这个 Python 解释器自己的绝对路径，不管
    这次是被 main.py 还是被装好的 steward 命令触发，指向的都是同一个真
    解释器。真正的 steward 入口脚本，跟这个解释器住在同一个 bin/ 目录下
    （pip install 生成的时候就是这么放的，见 docs/系统与运行时基础知识
    笔记.md 第四节），从这个解释器的路径推出 bin/ 目录，再找同目录下的
    steward，不管从哪种方式调用，得到的都是同一份、真正可执行的路径。

    注意：这里不能对 sys.executable 调用 .resolve()——venv 里的 python3
    通常是指向系统解释器的符号链接（验证过，见 docs/系统与运行时基础
    知识笔记.md 第七节），.resolve() 会一路穿透这条链接追到最底层的物理
    文件（比如 Homebrew Cellar 目录深处），那个目录底下根本没有 steward
    脚本——真实踩过：加了 .resolve() 之后在这台机器上实测直接报"找不到"。
    sys.executable 本身已经是绝对路径，不需要再 resolve 一次。
    """
    candidate = Path(sys.executable).parent / "steward"
    if not candidate.exists():
        raise RuntimeError(
            f"在 {candidate} 没找到 steward 入口脚本，"
            "先在这个环境里跑一遍 `pip install -e .`（或者装好对应 wheel）再重试。"
        )
    return str(candidate)


def setup_schedule(db_path=DEFAULT_DB_PATH):
    """生成/更新 launchd 的 WatchPaths 配置并重新加载，让 refresh 在数据库
    管理的目录发生变化时自动触发。

    WatchPaths 列表是静态写进 plist 文件的，不会自己感知数据库变化——每次
    新 `steward index` 了一个此前没管理过的新目录之后，需要重新跑这个函数
    （对应 CLI 的 `steward schedule-setup`），把新目录也纳入监听范围，这是
    显式的手动步骤，不是自动触发（原因见 product_direction_and_roadmap.md
    里的讨论：不想让 index 命令悄悄产生"改系统级 launchd 配置"这种跟它本身
    职责无关、还可能失败的副作用）。
    """
    with DocumentIndex(db_path) as index:
        target_dirs = index.list_target_dirs()

    if not target_dirs:
        print("数据库里还没有任何被管理的目录，先跑几次 `steward index <目录>` 再设置监听。")
        return

    try:
        steward_executable = _find_steward_executable()
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    stdout_path = REFRESH_LOG_PATH.parent / "refresh_stdout.log"
    stderr_path = REFRESH_LOG_PATH.parent / "refresh_stderr.log"
    plist_content = _build_plist_content(steward_executable, target_dirs, stdout_path, stderr_path)

    LAUNCHD_PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAUNCHD_PLIST_PATH.write_text(plist_content, encoding="utf-8")

    # 先 unload 再 load——launchd 不会因为文件内容变了就自动重新读取，必须
    # 显式让它重新加载一遍。第一次设置时 unload 会失败（还没 load 过），
    # 这是预期情况，忽略这个报错继续往下走 load，不能让它中断整个流程。
    subprocess.run(["launchctl", "unload", str(LAUNCHD_PLIST_PATH)], capture_output=True)
    result = subprocess.run(
        ["launchctl", "load", str(LAUNCHD_PLIST_PATH)], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ launchctl load 失败: {result.stderr.strip()}")
        return

    print(f"✅ 已注册/更新 launchd 监听任务，正在监听 {len(target_dirs)} 个目录：")
    for d in target_dirs:
        print(f"   - {d}")
    print(f"配置文件: {LAUNCHD_PLIST_PATH}")
