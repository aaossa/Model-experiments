import os

import GPUtil
import psutil
from humanize import naturalsize


def memory_report():
    # CPU RAM
    current_process = psutil.Process(os.getpid())
    print(f"Main process PID: {current_process.pid}")
    print(
        f"CPU RAM free: {naturalsize(psutil.virtual_memory().available)} | "
        f"Proc. size: {naturalsize(current_process.memory_info().rss)}"
    )
    for child in current_process.children(recursive=True):
        print(f">> Child {child.pid}: {naturalsize(child.memory_info().rss)}")
    # GPU RAM
    GPUs = GPUtil.getGPUs()
    if GPUs:
        gpu = GPUs[0]
        print(
            f"GPU RAM free: {gpu.memoryFree:.0f} MB | "
            f"Used: {gpu.memoryUsed:.0f} MB | "
            f"Util.: {gpu.memoryUtil*100:.0f}% | "
            f"Total: {gpu.memoryTotal:.0f} MB"
        )
    else:
        print("No GPU available")


if __name__ == '__main__':
    memory_report()
