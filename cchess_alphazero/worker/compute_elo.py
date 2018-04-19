import os
import sys
import shutil
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
from cchess_alphazero.lib.web_helper import http_request, download_file
from cchess_alphazero.lib.elo_helper import compute_elo

logger = getLogger(__name__)

def start(config: Config):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    m = Manager()
    response = http_request(config.internet.get_evaluate_model_url)
    while int(response['status']) == 0:
        data = response['data']
        logger.info(f"评测开始，基准模型：{data['base']['digest'][0:8]}, elo = {data['base']['elo']};"
                    f"待评测模型：{data['unchecked']['digest'][0:8]}, elo = {data['unchecked']['elo']}")
        # make path
        base_weight_path = os.path.join(config.resource.next_generation_model_dir, data['base']['digest'] + '.h5')
        ng_weight_path = os.path.join(config.resource.next_generation_model_dir, data['unchecked']['digest'] + '.h5')
        # load model
        model_base = load_model(config, base_weight_path, data['base']['digest'])
        model_ng = load_model(config, ng_weight_path, data['unchecked']['digest'])
        # make pipes
        model_base_pipes = m.list([model_base.get_pipes(need_reload=False) for _ in range(config.play.max_processes)])
        model_ng_pipes = m.list([model_ng.get_pipes(need_reload=False) for _ in range(config.play.max_processes)])

        # eval_worker = EvaluateWorker(config, model_base_pipes, model_ng_pipes, 0, data)
        # res = eval_worker.start()
        with ProcessPoolExecutor(max_workers=config.play.max_processes) as executor:
            futures = []
            for i in range(config.play.max_processes):
                eval_worker = EvaluateWorker(config, model_base_pipes, model_ng_pipes, pid=i, data=data)
                futures.append(executor.submit(eval_worker.start))
                sleep(1)
        
        wait(futures)
        # close pipe
        model_base.close_pipes()
        model_ng.close_pipes()
        # reset model
        model_base = None
        model_ng = None

        response = http_request(config.internet.get_evaluate_model_url)
    logger.info(f"没有待评测权重，请稍等或继续跑谱")

class EvaluateWorker:
    def __init__(self, config: Config, pipes1=None, pipes2=None, pid=None, data=None):
        self.config = config
        self.player_bt = None
        self.player_ng = None
        self.pid = pid
        self.pipes_bt = pipes1
        self.pipes_ng = pipes2
        self.data = data

    def start(self):
        logger.debug(f"Evaluate#Start Process index = {self.pid}, pid = {os.getpid()}")
        need_evaluate = True

        while need_evaluate:
            idx = 0 if random() > 0.5 else 1
            start_time = time()
            value, turns = self.start_game(idx)
            end_time = time()
            
            if (value == 1 and idx == 0) or (value == -1 and idx == 1):
                result = '基准模型胜'
            elif (value == 1 and idx == 1) or (value == -1 and idx == 0):
                result = '待评测模型胜'
            else:
                result = '双方连续60回合未吃子，和棋'

            if value == -1: # loss
                score = 0
            elif value == 1: # win
                score = 1
            else:
                score = 0.5

            if idx == 0:
                score = 1 - score
            else:
                score = score

            logger.info(f"进程{self.pid}评测完毕 用时{(end_time - start_time):.1f}秒, "
                         f"{turns / 2}回合, {result}, 得分：{score}, value = {value}, idx = {idx}")

            data = {'digest': self.data['unchecked']['digest'], 'score': score}
            response = http_request(self.config.internet.update_elo_url, post=True, data=data)
            if response and int(response['status']) == 0:
                logger.info('评测结果上传成功！')

            response = http_request(self.config.internet.get_evaluate_model_url)
            if int(response['status']) == 0 and response['data']['base']['digest'] == self.data['base']['digest']\
                and response['data']['unchecked']['digest'] == self.data['unchecked']['digest']:
                need_evaluate = True
                logger.info(f"进程{self.pid}继续评测")
            else:
                need_evaluate = False
                logger.info(f"进程{self.pid}终止评测")


    def start_game(self, idx):
        pipe1 = self.pipes_bt.pop()
        pipe2 = self.pipes_ng.pop()
        search_tree1 = defaultdict(VisitState)
        search_tree2 = defaultdict(VisitState)

        self.player1 = CChessPlayer(self.config, search_tree=search_tree1, pipes=pipe1, 
                        debugging=False, enable_resign=False)
        self.player2 = CChessPlayer(self.config, search_tree=search_tree2, pipes=pipe2, 
                        debugging=False, enable_resign=False)

        # even: bst = red, ng = black; odd: bst = black, ng = red
        if idx % 2 == 0:
            red = self.player1
            black = self.player2
            logger.info(f"进程id = {self.pid} 基准模型执红，待评测模型执黑")
        else:
            red = self.player2
            black = self.player1
            logger.info(f"进程id = {self.pid} 待评测模型执红，基准模型执黑")

        state = senv.INIT_STATE
        history = [state]
        value = 0       # best model's value
        turns = 0       # even == red; odd == black
        game_over = False
        no_eat_count = 0

        while not game_over:
            start_time = time()
            no_act = None
            if state in history[:-1]:
                no_act = []
                for i in range(len(history) - 1):
                    if history[i] == state:
                        no_act.append(history[i + 1])
            if turns % 2 == 0:
                action, _ = red.action(state, turns, no_act=no_act)
            else:
                action, _ = black.action(state, turns, no_act=no_act)
            end_time = time()
            if self.config.opts.log_move:
                logger.debug(f"进程id = {self.pid}, action = {action}, turns = {turns}, time = {(end_time-start_time):.1f}")
            if action is None:
                logger.debug(f"{turns % 2} (0 = red; 1 = black) has resigned!")
                value = -1
                break
            history.append(action)
            state, no_eat = senv.new_step(state, action)
            turns += 1
            if no_eat:
                no_eat_count += 1
            history.append(state)

            if no_eat_count >= 120:
                game_over = True
                value = 0
            else:
                game_over, value, final_move = senv.done(state)

        self.player1.close()
        self.player2.close()

        if turns % 2 == 1:  # black turn
            value = -value

        self.pipes_bt.append(pipe1)
        self.pipes_ng.append(pipe2)
        return value, turns


def load_model(config, weight_path, digest, config_file=None):
    model = CChessModel(config)
    if not config_file:
        config_path = config.resource.model_best_config_path
    else:
        config_path = os.path.join(config.resource.model_dir, config_file)
    if (not load_model_weight(model, config_path, weight_path)) or model.digest != digest:
        logger.info(f"开始下载权重 {digest[0:8]}")
        url = config.internet.download_base_url + digest + '.h5'
        download_file(url, weight_path)
        try:
            if not load_model_weight(model, config_path, weight_path):
                logger.info(f"待评测权重还未上传，请稍后再试")
                sys.exit()
        except ValueError as e:
            logger.error(f"权重架构不匹配，自动重新加载 {e}")
            return load_model(config, weight_path, digest, 'model_256f.json')
        except Exception as e:
            logger.error(f"加载权重发生错误：{e}，10s后自动重试下载")
            os.remove(weight_path)
            sleep(10)
            return load_model(config, weight_path, digest)
    logger.info(f"加载权重 {digest[0:8]} 成功")
    return model

