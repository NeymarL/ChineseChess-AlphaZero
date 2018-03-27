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

def start(config: Config):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    play = ObSelfPlay(config)
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
            action, policy = self.ai.action(state, self.env.num_halfmoves, no_act)
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
