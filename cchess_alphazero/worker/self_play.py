import os
from time import sleep
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
from threading import Thread
from time import time
from collections import defaultdict
from threading import Lock
from random import random

from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner
from cchess_alphazero.lib.data_helper import get_game_data_filenames, write_game_data_to_file
from cchess_alphazero.lib.model_helper import load_best_model_weight, save_as_best_model
from cchess_alphazero.lib.tf_util import set_session_config

logger = getLogger(__name__)

def load_model(config):
    model = CChessModel(config)
    if config.opts.new or not load_best_model_weight(model):
        model.build()
        save_as_best_model(model)
    return model

def start(config: Config):
    set_session_config(per_process_gpu_memory_fraction=0.5, allow_growth=True, device_list='1')
    current_model = load_model(config)
    m = Manager()
    cur_pipes = m.list([current_model.get_pipes(config.play.search_threads) \
                        for _ in range(config.play.max_processes)])

    # play_worker = SelfPlayWorker(config, cur_pipes)
    # play_worker.start()
    with ProcessPoolExecutor(max_workers=config.play.max_processes) as executor:
        futures = []
        for i in range(config.play.max_processes):
            play_worker = SelfPlayWorker(config, cur_pipes, i)
            logger.debug("Initialize selfplay worker")
            futures.append(executor.submit(play_worker.start))

class SelfPlayWorker:
    def __init__(self, config: Config, pipes=None, pid=None):
        self.config = config
        self.env = CChessEnv()
        self.red = None
        self.black = None
        self.buffer = []
        self.cur_pipes = pipes
        self.pid = pid

    def start(self):
        logger.debug(f"Selfplay#Start Process index = {self.pid}, pid = {os.getpid()}")

        self.buffer = []
        idx = 1

        while True:
            start_time = time()
            env = self.start_game(idx)
            end_time = time()
            logger.debug(f"Process{self.pid} play game {idx} time={end_time - start_time} sec, "
                         f"turn={env.num_halfmoves / 2}:{env.winner}")

            idx += 1
            # with open(self.config.resource.self_play_game_idx_file, "wt") as f:
            #     f.write(str(idx))

    def start_game(self, idx):
        pipes = self.cur_pipes.pop()
        env = CChessEnv().reset()
        search_tree = defaultdict(VisitState)

        self.red = CChessPlayer(self.config, search_tree=search_tree, pipes=pipes)
        self.black = CChessPlayer(self.config, search_tree=search_tree, pipes=pipes)

        history = []
        cc = 0

        while not env.done:
            start_time = time()
            if env.red_to_move:
                action = self.red.action(env)
            else:
                action = self.black.action(env)
            end_time = time()
            # logger.debug(f"Process{self.pid} Playing: {env.red_to_move}, action: {action}, time: {end_time - start_time}s")
            env.step(action)
            history.append(action)
            if len(history) > 6 and history[-1] == history[-5]:
                cc = cc + 1
            else:
                cc = 0
            if env.num_halfmoves / 2 >= self.config.play.max_game_length or cc >= 4:
                env.winner = Winner.draw
        if env.winner == Winner.red:
            red_win = 1
        elif env.winner == Winner.black:
            red_win = -1
        else:
            red_win = 0

        self.red.finish_game(red_win)
        self.black.finish_game(-red_win)

        self.cur_pipes.append(pipes)
        self.save_record_data(env, write=idx % self.config.play_data.nb_game_save_record == 0)
        self.save_play_data(write=idx % self.config.play_data.nb_game_in_file == 0)
        self.remove_play_data()
        return env

    def save_play_data(self, write=True):
        data = []
        for i in range(len(self.red.moves)):
            data.append(self.red.moves[i])
            if i < len(self.black.moves):
                data.append(self.black.moves[i])

        self.buffer += data

        if not write:
            return

        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"Process{self.pid} save play data to {path}")
        write_game_data_to_file(path, self.buffer)
        self.buffer = []

    def save_record_data(self, env, write=False):
        if not write:
            return
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_record_dir, rc.play_record_filename_tmpl % game_id)
        env.save_records(path)

    def remove_play_data(self):
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.play_data.max_file_num:
            return
        try:
            for i in range(len(files) - self.config.play_data.max_file_num):
                os.remove(files[i])
        except:
            pass

    def read_as_int(self, filename):
        if os.path.exists(filename):
            try:
                with open(filename, "rt") as f:
                    ret = int(str(f.read()).strip())
                    if ret:
                        return ret
            except ValueError:
                pass
