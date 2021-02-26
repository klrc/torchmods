import pynvml


def gpu_stat(index=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)  # 0表示显卡标号
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        'total': str(meminfo.total/1024**2)+'MiB',
        'used': str(meminfo.used/1024**2)+'MiB',
        'free': str(meminfo.free/1024**2)+'MiB',
    }
