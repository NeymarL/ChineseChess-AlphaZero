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
import subprocess

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
from cchess_alphazero.lib.web_helper import http_request

logger = getLogger(__name__)

def start(config: Config):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    base_model = {'digest': 'd6fce85e040a63966fa7651d4a08a7cdba2ef0e5975fc16a6d178c96345547b3', 'elo': 0}
    m = Manager()
    base_weight_path = os.path.join(config.resource.next_generation_model_dir, base_model['digest'] + '.h5')
    model_base = load_model(config, config.resource.model_best_config_path, base_weight_path)
    modelbt_pipes = m.list([model_base.get_pipes(need_reload=False) for _ in range(config.play.max_processes)])
    
    while True:
        while not check_ng_model(config, exculds=[base_model['digest'] + '.h5']):
            logger.info(f"Next generation model is None, wait for 300s")
            sleep(300)

        logger.info(f"Loading next generation model!")
        digest = check_ng_model(config, exculds=[base_model['digest'] + '.h5'])
        logger.debug(f"digest = {digest}")
        ng_weight_path = os.path.join(config.resource.next_generation_model_dir, digest)
        model_ng = load_model(config, config.resource.next_generation_config_path, ng_weight_path)
        modelng_pipes = m.list([model_ng.get_pipes(need_reload=False) for _ in range(config.play.max_processes)])

        # play_worker = EvaluateWorker(config, model1_pipes, model2_pipes)
        # play_worker.start()
        with ProcessPoolExecutor(max_workers=config.play.max_processes) as executor:
            futures = []
            for i in range(config.play.max_processes):
                eval_worker = EvaluateWorker(config, modelbt_pipes, modelng_pipes, pid=i)
                futures.append(executor.submit(eval_worker.start))
        
        wait(futures)
        model_base.close_pipes()
        model_ng.close_pipes()

        results = []
        for future in futures:
            results += future.result()
        base_elo = base_model['elo']
        ng_elo = base_elo
        for res in results:
            if res[1] == -1: # loss
                res[1] = 0
            elif res[1] != 1: # draw
                res[1] = 0.5
            if res[0] % 2 == 0:
                # red = base
                _, ng_elo = compute_elo(base_elo, ng_elo, res[1])
            else:
                # black = base
                ng_elo, _ = compute_elo(ng_elo, base_elo, 1 - res[1])
        logger.info(f"Evaluation finished, Next Generation's elo = {ng_elo}, base = {base_elo}")
        # send ng model to server
        logger.debug(f"Sending model to server")
        send_model(ng_weight_path)
        data = {'digest': digest[:-3], 'elo': ng_elo}
        http_request(config.internet.add_model_url, post=True, data=data)
        os.remove(base_weight_path)
        base_weight_path = ng_weight_path
        base_model['disgest'] = digest[:-3]
        base_model['elo'] = ng_elo
        model_base = model_ng
        modelbt_pipes = m.list([model_base.get_pipes(need_reload=False) for _ in range(config.play.max_processes)])


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
        results = []

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
            results.append((idx, score))

            logger.debug(f"Process{self.pid} play game {idx} time={(end_time - start_time):.1f} sec, "
                         f"turn={turns / 2}, model1 {score1} - {score2} model2")
        return results

    def start_game(self, idx):
        pipe1 = self.pipes_bt.pop()
        pipe2 = self.pipes_ng.pop()
        search_tree1 = defaultdict(VisitState)
        search_tree2 = defaultdict(VisitState)

        self.player1 = CChessPlayer(self.config, search_tree=search_tree1, pipes=pipe1, 
                        debugging=False, enable_resign=True)
        self.player2 = CChessPlayer(self.config, search_tree=search_tree2, pipes=pipe2, 
                        debugging=False, enable_resign=True)

        # even: bst = red, ng = black; odd: bst = black, ng = red
        if idx % 2 == 0:
            red = self.player1
            black = self.player2
            logger.debug(f"best model is red, ng is black")
        else:
            red = self.player2
            black = self.player1
            logger.debug(f"best model is black, ng is red")

        state = senv.INIT_STATE
        value = 0       # best model's value
        turns = 0       # even == red; odd == black
        game_over = False

        while not game_over:
            start_time = time()
            if turns % 2 == 0:
                action, _ = red.action(state, turns)
            else:
                action, _ = black.action(state, turns)
            end_time = time()
            # logger.debug(f"pid = {self.pid}, idx = {idx}, action = {action}, turns = {turns}, time = {(end_time-start_time):.1f}")
            if action is None:
                logger.debug(f"{turn % 2} (0 = red; 1 = black) has resigned!")
                value = -1
                break
            
            state = senv.step(state, action)
            turns += 1

            if turns / 2 >= self.config.play.max_game_length:
                game_over = True
                value = 0
            else:
                game_over, value, final_move = senv.done(state)

        self.player1.close()
        self.player2.close()

        if turns % 2 == 1:  # black turn
            value = -value

        if idx % 2 == 1:   # return player1' value
            value = -value

        self.pipes_bt.append(pipe1)
        self.pipes_ng.append(pipe2)
        return value, turns


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

def check_ng_model(config, exculds=[]):
    weights = [name for name in os.listdir(config.resource.next_generation_model_dir)
            if name.endswith('.h5')]
    for weight in weights:
        if weight not in exculds:
            return weight
    return None

def send_model(path):
    success = False
    for i in range(3):
        remote_server = 'root@115.159.183.150'
        remote_path = '/var/www/alphazero.52coding.com.cn/data/model'
        cmd = f'scp {path} {remote_server}:{remote_path}'
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode == 0:
            success = True
            logger.info("Send model success!")
            break
        else:
            logger.error(f"Send model failed! {ret.stderr}, {cmd}")
