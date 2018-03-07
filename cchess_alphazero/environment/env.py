import enum
import numpy as np
import copy

from cchess_alphazero.environment.chessboard import Chessboard
from cchess_alphazero.environment.lookup_tables import Chessman_2_idx

from logging import getLogger

logger = getLogger(__name__)

class ChessEnv:

    def __init__(self):
        self.board = None
        self.winner = None
        self.num_halfmoves = 0

    def reset(self):
        self.board = Chessboard()
        self.board.init_board()
        self.winner = None
        self.num_halfmoves = 0
        return self

    def update(self, board):
        self.board = board
        self.winner = None
        return self

    @property
    def done(self):
        return self.winner is not None

    @property
    def red_won(self):
        return self.winner == Winner.red

    @property
    def red_to_move(self):
        return self.board.is_red_turn

    @property
    def observation(self):
        return self.board.FENboard()

    def step(self, action: str, check_over = True):
        if check_over and action is None:
            return

        if not self.board.move_action_str(action):
            logger.error("Move Failed, action=%s, board=\n%s" % (action, self.board.screen))
        self.num_halfmoves += 1

        if check_over and self.board.is_end():
            self.winner = self.board.winner

    def copy(self):
        env = copy.deepcopy(self)
        env.board = copy.deepcopy(self.board)
        return env

    def render(self, gui = False):
        if gui:
            pass
        else:
            self.board.print_to_cl()

    def input_planes(self, flip = False):
        planes = np.zeros(shape=(14, 10, 9), dtype=np.float32)
        for chessman in self.board.chessmans_hash.values():
            point = chessman.position
            if not flip:
                planes[Chessman_2_idx[type(chessman)] + int(chessman.is_red) * 7][9 - point.y][point.x] = 1
            else:
                planes[Chessman_2_idx[type(chessman)] + int(not chessman.is_red) * 7][point.y][8 - point.x] = 1
        return planes


def test():
    env = ChessEnv()
    env.reset()
    print(env.board.legal_moves())
    print(env.observation)
    env.step('0001')
    env.render()
    print(env.board.legal_moves())
    print(env.observation)
    planes = env.input_planes()
    print("黑卒：\n", planes[0])
    print("红卒：\n",planes[7])
    print()
    inv_planes = env.input_planes(flip=True)
    print(inv_planes[2])

if __name__ == '__main__':
    test()

