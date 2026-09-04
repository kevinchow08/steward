"""资源监控:采样进程内存 RSS + CPU 占用,记录耗时。基于 psutil 实现,替代旧 Windows 上的 monitor_process.ps1。

GPU 算力占用（相当于"GPU 版的 CPU 占用率"）不测：macOS 上基本只能靠
powermetrics，这个命令需要 sudo，把一个要 sudo 的子进程调用嵌进日常跑的命令里，
意味着每次都可能被要求输入密码，这个体验代价要不要接受是个单独的决定，现阶段
不接受。

内存这块不受这个限制——Apple Silicon 是统一内存架构，GPU 用的显存和 CPU 用的
内存是同一块物理内存池，不需要专门的 GPU 工具，只要用 psutil 测对进程就行。
"测对进程"是关键：Week 3 打标签管线真正调用模型推理的是 llama-server 那个独立
进程，不是 steward 自己这个 Python 进程——两边的内存开销是两笔独立的账（steward
自己加载 embedding 模型、批量算向量矩阵是一笔，llama-server 加载模型权重、维护
KV cache 是另一笔），只测其中一个会丢另一半信息，所以 ResourceMonitor 支持传入
指定 pid，去监控任意一个本机进程，不是只能测"自己"。
"""

import time

import psutil


def find_process_by_name(name_substring):
    """按进程名找一个本机正在跑的进程，返回它的 pid；没找到返回 None。
    不需要 sudo——只是列出当前用户能看到的进程，不涉及提权。

    同名进程有多个时，返回找到的第一个——目前只用来找 llama-server 这种
    "正常情况下只会起一个"的服务，没有做"选哪一个"的消歧逻辑，也没必要。
    """
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if name_substring in proc.info["name"]:
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # 进程列表是那一瞬间的快照，遍历过程中进程退出、或者没权限看某个
            # 进程的详情，都不算错误，跳过继续找下一个就行。
            continue
    return None


class ResourceMonitor:
    def __init__(self, pid=None):
        """pid=None（默认）监控当前正在跑的这个进程本身；传一个具体的 pid，
        可以监控本机任意一个其它进程（比如 llama-server）——两种用法共用
        同一套采样/统计逻辑，不需要区分对待。
        """
        # psutil.Process() 不传参数默认指向"当前正在跑的这个进程"；传了 pid
        # 就指向那个进程。如果传入的 pid 已经不存在了（比如调用方拿到 pid 之后、
        # 真正创建监控之前，那个进程凑巧退出了），这里会抛 NoSuchProcess，
        # 交给调用方决定怎么处理（比如跳过这一路监控，不影响主流程）。
        self._process = psutil.Process(pid)

        # time.monotonic() 专门用来测"过了多久"这种场景:
        # 它是一个单调递增的计时器,不受系统时间被手动调整(比如切时区、校时)的影响,测耗时比 time.time() 更可靠
        self._start_time = time.monotonic()

        # memory_info().rss = Resident Set Size,进程当前实际占用的物理内存,单位是字节
        # 初始化时先采一次样,作为峰值的起点
        self._peak_rss = self._process.memory_info().rss

        # cpu_percent(interval=None) 测的是"上一次调用到这一次调用之间"这段时间的 CPU 占用率
        # 第一次调用没有"上一次"作参照,返回值没有意义,这里只是用来建立起始参照点,结果直接丢弃
        # 多核机器上占用率可能超过 100%(比如用满 2 个核就是 200%)
        self._process.cpu_percent(interval=None)
        self._peak_cpu_percent = 0.0

    def sample(self):
        # 监控的是别的进程时（比如 llama-server），采样过程中那个进程有可能
        # 已经退出了（服务被手动关掉、或者中途崩溃）——psutil 会抛
        # NoSuchProcess，这种情况不应该让整个采样/主流程跟着崩掉，安静地跳过
        # 这一次采样就行，之前已经采到的峰值数据依然保留、依然有效。
        try:
            current_rss = self._process.memory_info().rss
            if current_rss > self._peak_rss:
                self._peak_rss = current_rss

            current_cpu = self._process.cpu_percent(interval=None)
            if current_cpu > self._peak_cpu_percent:
                self._peak_cpu_percent = current_cpu
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def stop(self):
        self.sample()  # 结束前再采一次,避免漏掉最后一段的变化
        elapsed_seconds = time.monotonic() - self._start_time

        return {
            "elapsed_seconds": elapsed_seconds,
            "peak_rss_mb": self._peak_rss / (1024 * 1024),  # 字节转 MB,方便人读
            "peak_cpu_percent": self._peak_cpu_percent,
        }
