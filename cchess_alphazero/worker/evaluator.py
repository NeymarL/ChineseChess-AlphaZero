import os
import shutil
from time import sleep
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
from threading import Thread
from time import time, sleep
from collections import defaultdict
from multiprocessing import Lock
from random import random
import numpy as np

import cchess_alphazero.environment.static_env as senv
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

def start(config: Config):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    m = Manager()
    while True:
        model_bt = load_model(config, config.resource.model_best_config_path, config.resource.model_best_weight_path)
        modelbt_pipes = m.list([model_bt.get_pipes(need_reload=False) for _ in range(config.play.max_processes)])
        model_ng = load_model(config, config.resource.next_generation_config_path, config.resource.next_generation_weight_path)
        while not model_ng:
            logger.info(f"Next generation model is None, wait for 300s")
            sleep(300)
            model_ng = load_model(config, config.resource.next_generation_config_path, config.resource.next_generation_weight_path)
        logger.info(f"Next generation model has loaded!")
        modelng_pipes = m.list([model_ng.get_pipes(need_reload=False) for _ in range(config.play.max_processes)])

        # play_worker = EvaluateWorker(config, model1_pipes, model2_pipes)
        # play_worker.start()
        with ProcessPoolExecutor(max_workers=config.play.max_processes) as executor:
            futures = []
            for i in range(config.play.max_processes):
                eval_worker = EvaluateWorker(config, modelbt_pipes, modelng_pipes, pid=i)
                futures.append(executor.submit(eval_worker.start))
        
        wait(futures)
        model_bt.close_pipes()
        model_ng.close_pipes()
        # compute whether to update best model
        # and remove next generation model
        score = 0
        for future in futures:
            score += future.result()
        game_num = config.eval.game_num * config.play.max_processes
        logger.info(f"Evaluate over, next generation win {score}/{game_num}")
        if score * 1.0 / game_num >= config.eval.next_generation_replace_rate:
            logger.info("Best model will be replaced by next generation model")
            replace_best_model(config)
        else:
            logger.info("Next generation fail to defeat best model and will be removed")
            remove_ng_model(config)

class EvaluateWorker:
    def __init__(self, config: Config, pipes1=None, pipes2=None, pid=None):
        self.config = config
        self.player_bt = None
        self.player_ng = None
        self.pid = pid
        self.pipes_bt = pipes1
        self.pipes_ng = pipes2

    def start(self):
        logger.debug(f"Evaluate#Start Process index = {self.pid}, pid = {os.getpid()}")
        score1 = 0
        score2 = 0

        for idx in range(self.config.eval.game_num):
            start_time = time()
            score, turns = self.start_game(idx)
            end_time = time()

            if score < 0:
                score2 += 1
            elif score > 0:
                score1 += 1
            else:
                score2 += 0.5
                score1 += 0.5

            logger.debug(f"Process{self.pid} play game {idx} time={(end_time - start_time):.1f} sec, "
                         f"turn={turns / 2}, best model {score1} - {score2} next generation model")
        return score2  # return next generation model's score

    def start_game(self, idx):
        pipe1 = self.pipes_bt.pop()
        pipe2 = self.pipes_ng.pop()
        search_tree1 = defaultdict(VisitState)
        search_tree2 = defaultdict(VisitState)

        self.player1 = CChessPlayer(self.config, search_tree=search_tree1, pipes=pipe1, 
                        debugging=False, enable_resign=True)
        self.player2 = CChessPlayer(self.config, search_tree=search_tree2, pipes=pipe2, 
                        debugging=False, enable_resign=True)

        state = senv.INIT_STATE
        score = 0
        value = 0
        turns = 0       # even == red; odd == black
        game_over = False
        written = False

        while not game_over:
            # idx == 0 (even): player1 red; idx == 1 (odd): player2 red
            if turns % 2 == idx % 2:
                action, _ = self.player1.action(state, turns)
            else:
                action, _ = self.player2.action(state, turns)
            # logger.debug(f"pid = {self.pid}, idx = {idx}, action = {action}, turns = {turns}")
            if action is None:
                logger.debug(f"{turn % 2 == idx % 2} (1 = best model; 0 = next generation) has resigned!")
                if turn % 2 == idx % 2:
                    score = 1
                else:
                    score = -1
                written = True
                break
            
            state = senv.step(state, action)
            turns += 1

            if turns / 2 >= self.config.play.max_game_length:
                game_over = True
                score = 0.5
                written = True
            else:
                game_over, value = senv.done(state)

        self.player1.close()
        self.player2.close()

        if turns % 2 == 1:  # black turn
            value = -value

        if not written:
            if turns % 2 == idx % 2:
                # best model = red
                score = value
            else:
                # best model = black
                score = -value

        self.pipes_bt.append(pipe1)
        self.pipes_ng.append(pipe2)
        return score, turns


def replace_best_model(config):
    rc = config.resource
    shutil.copyfile(rc.next_generation_config_path, rc.model_best_config_path)
    shutil.copyfile(rc.next_generation_weight_path, rc.model_best_weight_path)
    remove_ng_model(config)

def remove_ng_model(config):
    rc = config.resource
    os.remove(rc.next_generation_config_path)
    os.remove(rc.next_generation_weight_path)

def load_model(config, config_path, weight_path, name=None):
    model = CChessModel(config)
    if not load_model_weight(model, config_path, weight_path, name):
        return None
    return model

