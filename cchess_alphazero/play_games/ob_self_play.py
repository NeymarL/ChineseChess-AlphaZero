import os
import subprocess
import numpy as np
from collections import defaultdict
from logging import getLogger
from time import sleep

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.environment.chessboard import Chessboard
from cchess_alphazero.environment.chessman import *
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_move
from cchess_alphazero.lib.model_helper import load_best_model_weight
from cchess_alphazero.lib.tf_util import set_session_config

logger = getLogger(__name__)

def start(config: Config, ucci=False, ai_move_first=True):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    if not ucci:
        play = ObSelfPlay(config)
    else:
        play = ObSelfPlayUCCI(config, ai_move_first)
    play.start()

class ObSelfPlay:
    def __init__(self, config: Config):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.chessmans = None

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def start(self):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=False)

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        self.env.board.print_to_cl()
        history = [self.env.get_state()]

        while not self.env.board.is_end():
            no_act = None
            state = self.env.get_state()
            if state in history[:-1]:
                no_act = []
                for i in range(len(history) - 1):
                    if history[i] == state:
                        no_act.append(history[i + 1])
            action, _ = self.ai.action(state, self.env.num_halfmoves, no_act)
            history.append(action)
            if action is None:
                print("AI投降了!")
                break
            move = self.env.board.make_single_record(int(action[0]), int(action[1]), int(action[2]), int(action[3]))
            if not self.env.red_to_move:
                action = flip_move(action)
            self.env.step(action)
            history.append(self.env.get_state())
            print(f"AI选择移动 {move}")
            self.env.board.print_to_cl()
            sleep(1)

        self.ai.close()
        print(f"胜者是 is {self.env.board.winner} !!!")
        self.env.board.print_record()

class ObSelfPlayUCCI:
    def __init__(self, config: Config, ai_move_first=True):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.chessmans = None
        self.ai_move_first = ai_move_first

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def start(self):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=False)

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        self.env.board.print_to_cl()
        history = [self.env.get_state()]
        turns = 0

        while not self.env.board.is_end():
            if (self.ai_move_first and turns % 2 == 0) or (not self.ai_move_first and turns % 2 == 1):
                no_act = None
                state = self.env.get_state()
                if state in history[:-1]:
                    no_act = []
                    for i in range(len(history) - 1):
                        if history[i] == state:
                            no_act.append(history[i + 1])
                action, _ = self.ai.action(state, self.env.num_halfmoves, no_act)
                if action is None:
                    print("AlphaZero 投降了!")
                    break
                move = self.env.board.make_single_record(int(action[0]), int(action[1]), int(action[2]), int(action[3]))
                print(f"AlphaZero 选择移动 {move}")
                if not self.env.red_to_move:
                    action = flip_move(action)
            else:
                state = self.env.get_state()
                fen = senv.state_to_fen(state, turns)
                print(f"fen = {fen}")
                action = self.get_ucci_move(fen)
                print(action)
                if not self.env.red_to_move:
                    rec_action = flip_move(action)
                move = self.env.board.make_single_record(int(rec_action[0]), int(rec_action[1]), int(rec_action[2]), int(rec_action[3]))
                print(f"Eleeye 选择移动 {move}")
            history.append(action)
            self.env.step(action)
            history.append(self.env.get_state())
            self.env.board.print_to_cl()
            turns += 1
            sleep(1)

        self.ai.close()
        print(f"胜者是 is {self.env.board.winner} !!!")
        self.env.board.print_record()

    def get_ucci_move(self, fen, time=3000):
        p = subprocess.Popen(self.config.resource.eleeye_path,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True)
        fen = f'position fen {fen}\n'
        cmd = 'ucci\n' + fen + f'go time {time}\n' + 'quit\n'
        out, err = p.communicate(cmd)
        print(cmd)
        print(out)
        lines = out.split('\n')
        move = lines[-3].split(' ')[1]
        print(move)
        return senv.parse_ucci_move(move)
