from collections import defaultdict
from logging import getLogger

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

logger = getLogger(__name__)

def start(config: Config, human_move_first=True):
    play = PlayWithHuman(config)
    play.start(human_move_first)

class PlayWithHuman:
    def __init__(self, config: Config):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.chessmans = None
        self.human_move_first = True

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def start(self, human_first=True):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=False)
        self.human_move_first = human_first

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        self.env.board.print_to_cl()

        while not self.env.board.is_end():
            if human_first == self.env.red_to_move:
                self.env.board.calc_chessmans_moving_list()
                is_correct_chessman = False
                is_correct_position = False
                chessman = None
                while not is_correct_chessman:
                    title = "请输入棋子位置: "
                    input_chessman_pos = input(title)
                    x, y = int(input_chessman_pos[0]), int(input_chessman_pos[1])
                    chessman = self.env.board.chessmans[x][y]
                    if chessman != None and chessman.is_red == self.env.board.is_red_turn:
                        is_correct_chessman = True
                        print(f"当前棋子为{chessman.name_cn}，可以落子的位置有：")
                        for point in chessman.moving_list:
                            print(point.x, point.y)
                    else:
                        print("没有找到此名字的棋子或未轮到此方走子")
                while not is_correct_position:
                    title = "请输入落子的位置: "
                    input_chessman_pos = input(title)
                    x, y = int(input_chessman_pos[0]), int(input_chessman_pos[1])
                    is_correct_position = chessman.move(x, y)
                    if is_correct_position:
                        self.env.board.print_to_cl()
                        self.env.board.clear_chessmans_moving_list()
            else:
                action, policy = self.ai.action(self.env.get_state(), self.env.num_halfmoves)
                if not self.env.red_to_move:
                    action = flip_move(action)
                if action is None:
                    print("AI投降了!")
                    break
                self.env.step(action)
                print(f"AI选择移动 {action}")
                self.env.board.print_to_cl()

        self.ai.close()
        print(f"胜者是 is {self.env.board.winner} !!!")
        self.env.board.print_record()
