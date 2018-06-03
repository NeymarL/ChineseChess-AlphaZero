import os
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
from random import random, randint
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
    # while True:
    model_bt = load_model(config, config.resource.model_best_config_path, config.resource.model_best_weight_path)
    modelbt_pipes = m.list([model_bt.get_pipes(need_reload=False) for _ in range(config.play.max_processes)])
    model_ng = load_model(config, config.resource.next_generation_config_path, config.resource.next_generation_weight_path)
    # while not model_ng:
    #     logger.info(f"Next generation model is None, wait for 300s")
    #     sleep(300)
    #     model_ng = load_model(config, config.resource.next_generation_config_path, config.resource.next_generation_weight_path)
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
    total_score = 0
    red_new_win = 0
    red_new_fail = 0
    red_new_draw = 0
    black_new_win = 0
    black_new_fail = 0
    black_new_draw = 0
    for future in futures:
        data = future.result()
        total_score += data[0]
        red_new_win += data[1]
        red_new_draw += data[2]
        red_new_fail += data[3]
        black_new_win += data[4]
        black_new_draw += data[5]
        black_new_fail += data[6]
    game_num = config.eval.game_num * config.play.max_processes
    win_rate = total_score * 100 / game_num
    logger.info(f"Evaluate over, next generation win {total_score}/{game_num} = {win_rate:.2f}%")
    logger.info(f"红\t黑\t胜\t平\t负")
    logger.info(f"新\t旧\t{red_new_win}\t{red_new_draw}\t{red_new_fail}")
    logger.info(f"旧\t新\t{black_new_win}\t{black_new_draw}\t{black_new_fail}")
    # if total_score * 1.0 / game_num >= config.eval.next_generation_replace_rate:
    #     logger.info("Best model will be replaced by next generation model")
    #     replace_best_model(config)
    # else:
    #     logger.info("Next generation fail to defeat best model and will be removed")
    #     remove_ng_model(config)

class EvaluateWorker:
    def __init__(self, config: Config, pipes1=None, pipes2=None, pid=None):
        self.config = config
        self.player_bt = None
        self.player_ng = None
        self.pid = pid
        self.pipes_bt = pipes1
        self.pipes_ng = pipes2

    def start(self):
        ran = self.config.play.max_processes * 2
        sleep((self.pid % ran) * 10)
        logger.debug(f"Evaluate#Start Process index = {self.pid}, pid = {os.getpid()}")
        score = 0
        total_score = 0
        red_new_win = 0
        red_new_fail = 0
        red_new_draw = 0
        black_new_win = 0
        black_new_fail = 0
        black_new_draw = 0

        for idx in range(self.config.eval.game_num):
            start_time = time()
            value, turns = self.start_game(idx)
            end_time = time()

            if (value == 1 and idx % 2 == 0) or (value == -1 and idx % 2 == 1):
                if idx % 2 == 0:
                    black_new_fail += 1
                else:
                    red_new_fail += 1
                result = '基准模型胜'
            elif (value == 1 and idx % 2 == 1) or (value == -1 and idx % 2 == 0):
                if idx % 2 == 0:
                    black_new_win += 1
                else:
                    red_new_win += 1
                result = '待评测模型胜'
            else:
                if idx % 2 == 0:
                    black_new_draw += 1
                else:
                    red_new_draw += 1
                result = '和棋'

            if value == -1: # loss
                score = 0
            elif value == 1: # win
                score = 1
            else:
                score = 0.5

            if idx % 2 == 0:
                score = 1 - score
            else:
                score = score

            logger.info(f"进程{self.pid}评测完毕 用时{(end_time - start_time):.1f}秒, "
                         f"{turns / 2}回合, {result}, 得分：{score}, value = {value}, idx = {idx}")
            total_score += score
        return (total_score, red_new_win, red_new_draw, red_new_fail, black_new_win, black_new_draw, black_new_fail)

    def start_game(self, idx):
        pipe1 = self.pipes_bt.pop()
        pipe2 = self.pipes_ng.pop()
        search_tree1 = defaultdict(VisitState)
        search_tree2 = defaultdict(VisitState)

        playouts = randint(8, 12) * 100
        self.config.play.simulation_num_per_move = playouts
        logger.info(f"Set playouts = {self.config.play.simulation_num_per_move}")

        self.player1 = CChessPlayer(self.config, search_tree=search_tree1, pipes=pipe1, 
                        debugging=False, enable_resign=False)
        self.player2 = CChessPlayer(self.config, search_tree=search_tree2, pipes=pipe2, 
                        debugging=False, enable_resign=False)

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
        history = [state]
        value = 0       # best model's value
        turns = 0       # even == red; odd == black
        game_over = False
        no_eat_count = 0
        check = False

        while not game_over:
            start_time = time()
            no_act = None
            increase_temp = False
            if not check and state in history[:-1]:
                no_act = []
                increase_temp = True
                free_move = defaultdict(int)
                for i in range(len(history) - 1):
                    if history[i] == state:
                        # 如果走了下一步是将军或捉：禁止走那步
                        if senv.will_check_or_catch(state, history[i+1]):
                            no_act.append(history[i + 1])
                        # 否则当作闲着处理
                        else:
                            free_move[state] += 1
                            if free_move[state] >= 3:
                                # 作和棋处理
                                game_over = True
                                value = 0
                                logger.info("闲着循环三次，作和棋处理")
                                break
            if game_over:
                break
            if turns % 2 == 0:
                action, _ = red.action(state, turns, no_act=no_act, increase_temp=increase_temp)
            else:
                action, _ = black.action(state, turns, no_act=no_act, increase_temp=increase_temp)
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
            else:
                no_eat_count = 0
            history.append(state)

            if no_eat_count >= 120 or turns / 2 >= self.config.play.max_game_length:
                game_over = True
                value = 0
            else:
                game_over, value, final_move, check = senv.done(state, need_check=True)
                if not game_over:
                    if not senv.has_attack_chessman(state):
                        logger.info(f"双方无进攻子力，作和。state = {state}")
                        game_over = True
                        value = 0

        if final_move:
            history.append(final_move)
            state = senv.step(state, final_move)
            turns += 1
            value = - value
            history.append(state)

        self.player1.close()
        self.player2.close()

        if turns % 2 == 1:  # black turn
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

