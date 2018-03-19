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
from multiprocessing import Lock
from random import random
import numpy as np

from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, flip_move, ActionLabelsRed
from cchess_alphazero.lib.data_helper import get_game_data_filenames, write_game_data_to_file
from cchess_alphazero.lib.model_helper import load_model_weight
from cchess_alphazero.lib.tf_util import set_session_config

logger = getLogger(__name__)

def load_model(config, config_path, weight_path, name=None):
    model = CChessModel(config)
    load_model_weight(model, config_path, weight_path, name)
    return model

def start(config: Config):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list='0,1')
    model1 = load_model(config, config.resource.eval_model1_config_path, config.resource.eval_model1_weight_path, 
                        config.resource.model1_name)
    model2 = load_model(config, config.resource.eval_model2_config_path, config.resource.eval_model2_weight_path, 
                        config.resource.model2_name)
    m = Manager()
    model1_pipes = m.list([model1.get_pipes(config.play.search_threads) \
                        for _ in range(config.play.max_processes)])
    model2_pipes = m.list([model2.get_pipes(config.play.search_threads) \
                        for _ in range(config.play.max_processes)])

    # play_worker = EvaluateWorker(config, model1_pipes, model2_pipes)
    # play_worker.start()
    with ProcessPoolExecutor(max_workers=config.play.max_processes) as executor:
        futures = []
        for i in range(config.play.max_processes):
            eval_worker = EvaluateWorker(config, model1_pipes, model2_pipes, i)
            futures.append(executor.submit(eval_worker.start))

class EvaluateWorker:
    def __init__(self, config: Config, pipes1=None, pipes2=None, pid=None):
        self.config = config
        self.player1 = None
        self.player2 = None
        self.pid = pid
        self.pipes1 = pipes1
        self.pipes2 = pipes2

    def start(self):
        logger.debug(f"Evaluate#Start Process index = {self.pid}, pid = {os.getpid()}")
        score1 = 0
        score2 = 0

        for idx in range(self.config.eval.game_num):
            start_time = time()
            env, score = self.start_game(idx)
            end_time = time()

            if score < 0:
                score2 += 1
            elif score > 0:
                score1 += 1
            else:
                score2 += 0.5
                score1 += 0.5

            logger.debug(f"Process{self.pid} play game {idx} time={end_time - start_time} sec, "
                         f"turn={env.num_halfmoves / 2}, {self.config.resource.model1_name}:{self.config.resource.model2_name}"
                         f" = {score1}-{score2}")
        logger.info(f"Final result: {self.config.resource.model1_name} {score1} - {score2} {self.config.resource.model2_name}")


    def start_game(self, idx):
        pipe1 = self.pipes1.pop()
        pipe2 = self.pipes2.pop()
        search_tree1 = defaultdict(VisitState)
        search_tree2 = defaultdict(VisitState)

        env = CChessEnv(self.config).reset()

        self.player1 = CChessPlayer(self.config, search_tree=search_tree1, pipes=pipe1, debugging=False)
        self.player2 = CChessPlayer(self.config, search_tree=search_tree2, pipes=pipe2, debugging=False)

        history = []
        cc = 0

        while not env.done:
            start_time = time()
            # idx == 0 (even): player1 red; idx == 1 (odd): player2 red
            if int(env.red_to_move) == idx % 2:
                action = self.player2.action(env)
                end_time = time()
                # --------------------- debug logs ---------------------------
                # if not env.red_to_move:
                #     move = flip_move(action)
                # else:
                #     move = action
                # move = env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
                # logger.debug(f"Process{self.pid} Player2 action: {move}, time: {end_time - start_time}s")
                # key = self.player2.get_state_key(env)
                # p, v = self.player2.debug[key]
                # mov_idx = np.argmax(p)
                # move = ActionLabelsRed[mov_idx]
                # move = env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
                # logger.debug(f"P2 NN recommend move: {move} with probability {np.max(p)}, v = {v}")
                # logger.info("MCTS results:")
                # for move, action_state in self.player2.search_results.items():
                #     if action_state[0] >= 5:
                #         move = env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
                #         logger.info(f"move: {move}, prob: {action_state[0]}, Q_value: {action_state[1]}")
                # self.player2.search_results = {}
                # --------------------- debug logs ---------------------------
            else:
                action = self.player1.action(env)
                end_time = time()
                # --------------------- debug logs ---------------------------
                # if not env.red_to_move:
                #     move = flip_move(action)
                # else:
                #     move = action
                # move = env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
                # logger.debug(f"Process{self.pid} Player1 action: {move}, time: {end_time - start_time}s")
                # key = self.player1.get_state_key(env)
                # p, v = self.player1.debug[key]
                # mov_idx = np.argmax(p)
                # move = ActionLabelsRed[mov_idx]
                # move = env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
                # logger.debug(f"P1 NN recommend move: {move} with probability {np.max(p)}, v = {v}")
                # logger.info("MCTS results:")
                # for move, action_state in self.player1.search_results.items():
                #     if action_state[0] >= 5:
                #         move = env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
                #         logger.info(f"move: {move}, prob: {action_state[0]}, Q_value: {action_state[1]}")
                # self.player1.search_results = {}
                # --------------------- debug logs ---------------------------
            
            env.step(action)

            history.append(action)
            if len(history) > 6 and history[-1] == history[-5]:
                cc = cc + 1
            else:
                cc = 0
            if env.num_halfmoves / 2 >= self.config.play.max_game_length:
                env.winner = Winner.draw

        if env.winner == Winner.red:
            if idx % 2 == 0:
                p1_win = 1
            else:
                p1_win = -1
        elif env.winner == Winner.black:
            if idx % 2 == 0:
                p1_win = -1
            else:
                p1_win = 1
        else:
            p1_win = 0

        self.pipes1.append(pipe1)
        self.pipes2.append(pipe2)
        self.save_record_data(env, write=idx % self.config.play_data.nb_game_save_record == 0)
        return env, p1_win

    def save_record_data(self, env, write=False):
        if not write:
            return
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_record_dir, rc.play_record_filename_tmpl % game_id)
        env.save_records(path)



