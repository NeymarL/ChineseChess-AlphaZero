import os
import sys
import multiprocessing as mp

from logging import getLogger

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

from cchess_alphazero.lib.logger import setup_logger
from cchess_alphazero.config import Config, PlayWithHumanConfig
import cchess_alphazero.worker.self_play_windows as self_play

def setup_parameters(config):
    username = input(f"请输入用户名：")
    config.internet.username = username
    gpu = input(f"请输入GPU编号（0代表第一块，1代表第二块，以此类推...）：")
    config.opts.device_list = gpu
    num_cores = mp.cpu_count()
    max_processes = num_cores // 2 if num_cores < 20 else 10
    search_threads = 20
    max_processes = input(f"请输入运行进程数（推荐{max_processes}）：")
    max_processes = int(max_processes)
    print(f"max_processes = {max_processes}, search_threads = {search_threads}")
    config.play.max_processes = max_processes
    config.play.search_threads = search_threads


if __name__ == "__main__":
    mp.freeze_support()
    sys.setrecursionlimit(10000)
    config_type = 'distribute'
    config = Config(config_type=config_type)
    config.resource.create_directories()
    setup_logger(config.resource.main_log_path)
    config.internet.distributed = True
    setup_parameters(config)
    self_play.start(config)
