import os
from time import sleep
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
from time import time, sleep
from collections import defaultdict
from random import random

import cchess_alphazero.environment.static_env as senv
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
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list='0,1')
    current_model = load_model(config)
    m = Manager()
    cur_pipes = m.list([current_model.get_pipes() for _ in range(config.play.max_processes)])

    play_worker = SelfPlayWorker(config, cur_pipes, 0)
    play_worker.start()
    # with ProcessPoolExecutor(max_workers=config.play.max_processes) as executor:
    #     futures = []
    #     for i in range(config.play.max_processes):
    #         play_worker = SelfPlayWorker(config, cur_pipes, i)
    #         logger.debug("Initialize selfplay worker")
    #         futures.append(executor.submit(play_worker.start))

class SelfPlayWorker:
    def __init__(self, config: Config, pipes=None, pid=None):
        self.config = config
        self.player = None
        self.cur_pipes = pipes
        self.pid = pid
        self.buffer = []

    def start(self):
        logger.debug(f"Selfplay#Start Process index = {self.pid}, pid = {os.getpid()}")

        idx = 1
        self.buffer = []
        search_tree = defaultdict(VisitState)

        while True:
            start_time = time()
            value, turns, state, search_tree = self.start_game(idx, search_tree)
            end_time = time()
            logger.debug(f"Process{self.pid} play game {idx} time={(end_time - start_time):.1f} sec, "
                         f"turn={turns / 2}, winner = {value} (1 = red, -1 = black, 0 draw)")
            if turns <= 10:
                senv.render(state)

            idx += 1

    def start_game(self, idx, search_tree):
        pipes = self.cur_pipes.pop()

        if not self.config.play.share_mtcs_info_in_self_play or \
            idx % self.config.play.reset_mtcs_info_per_game == 0:
            search_tree = defaultdict(VisitState)

        if random() > self.config.play.enable_resign_rate:
            enable_resign = True
            logger.debug(f"game {idx} enable resign!")
        else:
            enable_resign = False
            logger.debug(f"game {idx} disable resign!")

        self.player = CChessPlayer(self.config, search_tree=search_tree, pipes=pipes, enable_resign=enable_resign, debugging=True)

        state = senv.INIT_STATE
        history = [state]
        policys = [] 
        value = 0
        turns = 0       # even == red; odd == black
        game_over = False

        while not game_over:
            start_time = time()
            action, policy = self.player.action(state, turns)
            end_time = time()
            if action is None:
                logger.debug(f"{turn % 2} (0 = red; 1 = black) has resigned!")
                break
            logger.debug(f"Process{self.pid} Playing: {turns % 2}, action: {action}, time: {(end_time - start_time):.1f}s")
            # for move, action_state in self.player.search_results.items():
            #     if action_state[0] >= 20:
            #         logger.info(f"move: {move}, prob: {action_state[0]}, Q_value: {action_state[1]:.2f}, Prior: {action_state[2]:.3f}")
            self.player.search_results = {}
            policys.append(policy)
            state = senv.step(state, action)
            turns += 1
            history.append(state)

            if turns / 2 >= self.config.play.max_game_length:
                game_over = True
                value = 0
            else:
                game_over, value = senv.done(state)

        self.player.close()
        if turns % 2 == 1:  # balck turn
            value = -value

        v = value
        data = []
        for i in range(turns):
            data.append([history[i], policys[i], value])
            value = -value

        self.cur_pipes.append(pipes)
        self.save_play_data(idx, data)
        self.remove_play_data()
        return value, turns, state, search_tree

    def save_play_data(self, idx, data):
        self.buffer += data

        if not idx % self.config.play_data.nb_game_in_file == 0:
            return

        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"Process {self.pid} save play data to {path}")
        write_game_data_to_file(path, self.buffer)
        self.buffer = []

    def remove_play_data(self):
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.play_data.max_file_num:
            return
        try:
            for i in range(len(files) - self.config.play_data.max_file_num):
                os.remove(files[i])
        except:
            pass

