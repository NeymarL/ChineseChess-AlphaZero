import os
import sys
import multiprocessing as mp

from logging import getLogger

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

from cchess_alphazero.lib.logger import setup_logger
from cchess_alphazero.config import Config, PlayWithHumanConfig
from cchess_alphazero.worker import self_play

def setup_parameters(config):
    num_cores = mp.cpu_count()
    max_processes = num_cores // 2 if num_cores < 20 else 10
    search_threads = 10 if num_cores < 10 else (num_cores // 10) * 10
    print(f"max_processes = {max_processes}, search_threads = {search_threads}")
    config.play.max_processes = max_processes
    config.play.search_threads = search_threads

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    config_type = 'distribute'
    config = Config(config_type=config_type)
    config.device_list = '0'
    config.resource.create_directories()
    setup_logger(config.resource.main_log_path)
    config.internet.distributed = True
    config.opts.log_move = True
    setup_parameters(config)
    self_play.start(config)
