"""资源监控:采样进程内存 RSS,记录耗时。基于 psutil 实现,替代旧 Windows 上的 monitor_process.ps1。"""


class ResourceMonitor:
    def __init__(self):
        # TODO: 记录起始时间、初始化 psutil.Process()
        raise NotImplementedError

    def stop(self):
        # TODO: 记录结束时间,采样峰值 RSS,返回 {elapsed_seconds, peak_rss_mb, ...}
        raise NotImplementedError
