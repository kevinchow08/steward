"""资源监控:采样进程内存 RSS + CPU 占用,记录耗时。基于 psutil 实现,替代旧 Windows 上的 monitor_process.ps1。

GPU 占用现在不测:Week 1 不涉及模型推理,pipeline 不会碰 GPU/ANE,测了也是噪声。
等 Week 2 接模型之后 GPU 才是真正要回答的问题,到时候再考虑怎么测(psutil 不支持,得靠 powermetrics)。
"""

import time

import psutil


class ResourceMonitor:
    def __init__(self):
        # psutil.Process() 不传参数默认指向"当前正在跑的这个进程"(也就是跑 main.py 的这个 Python 进程本身)
        self._process = psutil.Process()

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
        # 每调一次,就检查当前值有没有超过历史峰值,超过了就更新
        # 之所以要"多次采样取峰值"而不是只在最后测一次,是因为占用中途可能有波峰,跑完再测会错过
        current_rss = self._process.memory_info().rss
        if current_rss > self._peak_rss:
            self._peak_rss = current_rss

        current_cpu = self._process.cpu_percent(interval=None)
        if current_cpu > self._peak_cpu_percent:
            self._peak_cpu_percent = current_cpu

    def stop(self):
        self.sample()  # 结束前再采一次,避免漏掉最后一段的变化
        elapsed_seconds = time.monotonic() - self._start_time

        return {
            "elapsed_seconds": elapsed_seconds,
            "peak_rss_mb": self._peak_rss / (1024 * 1024),  # 字节转 MB,方便人读
            "peak_cpu_percent": self._peak_cpu_percent,
        }
