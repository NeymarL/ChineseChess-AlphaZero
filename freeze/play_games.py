import os
import sys
import multiprocessing as mp

from logging import getLogger

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)


from cchess_alphazero.lib.logger import setup_logger
from cchess_alphazero.config import Config, PlayWithHumanConfig
from cchess_alphazero.play_games import play


def setup_parameters(config):
    num_cores = mp.cpu_count()
    search_threads = 10 if num_cores < 10 else (num_cores // 10) * 10
    print(f"search_threads = {search_threads}")
    config.play.search_threads = search_threads

if __name__ == "__main__":
    mp.freeze_support()
    sys.setrecursionlimit(10000)
    config_type = 'distribute'

    config = Config(config_type=config_type)
    config.opts.device_list = '0'
    config.resource.create_directories()
    setup_logger(config.resource.play_log_path)
    config.opts.new = False
    config.opts.light = False
    pwhc = PlayWithHumanConfig()
    pwhc.update_play_config(config.play)
    config.opts.bg_style = 'WOOD'
    setup_parameters(config)
    simulation_num = input('请输入AI搜索次数（必须为整数）：')
    ai_move_first = input('AI执红？(Y/N)')
    ai_move_first = True if ai_move_first == 'Y' or ai_move_first == 'y' else False
    config.play.simulation_num_per_move = int(simulation_num)
    play.start(config, not ai_move_first)
    input('按任意键退出...')
