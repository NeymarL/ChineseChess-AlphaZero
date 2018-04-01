import os
import subprocess
import numpy as np
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
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, flip_policy, flip_move
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
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    current_model = load_model(config)
    m = Manager()
    cur_pipes = m.list([current_model.get_pipes() for _ in range(config.play.max_processes)])

    # play_worker = SelfPlayWorker(config, cur_pipes, 0)
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
        self.player = None
        self.cur_pipes = pipes
        self.id = pid
        self.buffer = []
        self.pid = os.getpid()

    def start(self):
        logger.debug(f"Selfplay#Start Process index = {self.id}, pid = {self.pid}")

        idx = 1
        self.buffer = []
        search_tree = defaultdict(VisitState)

        while True:
            start_time = time()
            value, turns, state, search_tree, store = self.start_game(idx, search_tree)
            end_time = time()
            if value != 1 and value != -1:
                winner = 'Draw'
            elif idx % 2 == 0 and value == 1 or idx % 2 == 1 and value == -1:
                winner = 'AlphaHe'
            else:
                winner = 'Eleeye'

            logger.debug(f"Process {self.pid}-{self.id} play game {idx} time={(end_time - start_time):.1f} sec, "
                         f"turn={turns / 2}, value = {value:.2f}, winner is {winner}")
            if turns <= 10:
                senv.render(state)
            if store:
                idx += 1

    def start_game(self, idx, search_tree):
        pipes = self.cur_pipes.pop()

        if not self.config.play.share_mtcs_info_in_self_play or \
            idx % self.config.play.reset_mtcs_info_per_game == 0:
            search_tree = defaultdict(VisitState)

        if random() > self.config.play.enable_resign_rate:
            enable_resign = True
        else:
            enable_resign = False

        self.player = CChessPlayer(self.config, search_tree=search_tree, pipes=pipes, enable_resign=enable_resign, debugging=False)

        state = senv.INIT_STATE
        history = [state]
        policys = [] 
        value = 0
        turns = 0       # even == red; odd == black
        game_over = False
        is_alpha_red = True if idx % 2 == 0 else False

        while not game_over:
            if (is_alpha_red and turns % 2 == 0) or (not is_alpha_red and turns % 2 == 1):
                no_act = None
                if state in history[:-1]:
                    no_act = []
                    for i in range(len(history) - 1):
                        if history[i] == state:
                            no_act.append(history[i + 1])
                action, policy = self.player.action(state, turns, no_act)
                if action is None:
                    logger.debug(f"{turns % 2} (0 = red; 1 = black) has resigned!")
                    value = -1
                    break
            else:
                fen = senv.state_to_fen(state, turns)
                action = self.get_ucci_move(fen)
                if action is None:
                    logger.debug(f"{turns % 2} (0 = red; 1 = black) has resigned!")
                    value = -1
                    break
                if turns % 2 == 1:
                    action = flip_move(action)
                try:
                    policy = self.build_policy(action, False)
                except Exception as e:
                    logger.error(f"Build policy error {e}, action = {action}, state = {state}, fen = {fen}")
                    value = 0
                    break
            history.append(action)
            policys.append(policy)
            state = senv.step(state, action)
            turns += 1
            history.append(state)

            if turns / 2 >= self.config.play.max_game_length:
                game_over = True
                value = senv.evaluate(state)
            else:
                game_over, value = senv.done(state)

        if turns < 8:
            logger.debug(f"history = {history}")

        self.player.close()
        if turns % 2 == 1:  # balck turn
            value = -value

        v = value
        if v == 0:
            if random() > 0.5:
                store = True
            else:
                store = False
        else:
            store = True

        if store:
            data = []
            for i in range(turns):
                k = i * 2
                data.append([history[k], policys[i], value])
                value = -value
            self.save_play_data(idx, data)

        self.cur_pipes.append(pipes)
        self.remove_play_data()
        return v, turns, state, search_tree, store

    def get_ucci_move(self, fen, time=3):
        p = subprocess.Popen(self.config.resource.eleeye_path,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True)
        setfen = f'position fen {fen}\n'
        setrandom = 'setoption randomness small\n'
        cmd = 'ucci\n' + setrandom + setfen + f'go time {time * 1000}\n'
        try:
            out, err = p.communicate(cmd, timeout=time+0.5)
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                out, err = p.communicate()
            except Exception as e:
                logger.error(f"{e}, cmd = {cmd}")
                return self.get_ucci_move(fen, time+1)
        lines = out.split('\n')
        if lines[-2] == 'nobestmove':
            return None
        move = lines[-2].split(' ')[1]
        if move == 'depth':
            move = lines[-1].split(' ')[6]
        return senv.parse_ucci_move(move)

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

    def build_policy(self, action, flip):
        labels_n = len(ActionLabelsRed)
        move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
        policy = np.zeros(labels_n)

        policy[move_lookup[action]] = 1

        if flip:
            policy = flip_policy(policy)
        return list(policy)

