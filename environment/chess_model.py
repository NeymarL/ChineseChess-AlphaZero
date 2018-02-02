#-*- coding:utf-8 -*-

import torch
import chessboard
import chessman
from lookup_tables import *

class Observation(object):

    def __init__(self, board):
        self.__red = torch.FloatTensor(7*8*2+1, 10, 9).zero_()     # features to feed into the agent that play red
        self.__black = torch.FloatTensor(7*8*2+1, 10, 9).zero_()   # # features to feed into the agent that play black
        self.board = board
        self.t = 0
        self.make_features()

    @property
    def black(self):
        return self.__black

    @property
    def red(self):
        return self.__red

    def make_features(self):
        '''
        格式[2*8*7+1, 10, 9]：
            我方[8*7, 10, 9]：
                t = T: 车马炮等位置(7*10*9)
                    ...
                t = T-7: 车马炮等位置(7*10*9)
            对方[8*7, 10, 9]：
                t = T: 车马炮等位置(7*10*9)
                    ...
                t = T-7: 车马炮等位置(7*10*9)
            颜色[1, 10, 9]：
                红/黑 (10*9)
        '''
        # 我方 [0:7] = 0, 1, 2, 3, 4, 5, 6
        self.__red[0:49] = self.__red[7:56]
        self.__red[49:56] = torch.FloatTensor(7, 10, 9).zero_()
        # 对方
        self.__red[56:105] = self.__red[63:112]
        self.__red[105:112] = torch.FloatTensor(7, 10, 9).zero_()
        
        # 我方 [0:7] = 0, 1, 2, 3, 4, 5, 6
        self.__black[0:49] = self.__black[7:56]
        self.__black[49:56] = torch.FloatTensor(7, 10, 9).zero_()
        # 对方
        self.__black[56:105] = self.__black[63:112]
        self.__black[105:112] = torch.FloatTensor(7, 10, 9).zero_()
        
        # t = T
        for i in xrange(0, 9):
            for j in xrange(0, 10):
                chess = self.board.chessmans[i][j]
                if chess != None:
                    if chess.is_red:
                        # 我方 = 红
                        self.__red[49 + Chessman_2_idx[type(chess)]][9-j][i] = 1
                        # 对方
                        self.__black[105 + Chessman_2_idx[type(chess)]][j][8-i] = 1
                    else:
                        # 我方 = 黑
                        self.__black[49 + Chessman_2_idx[type(chess)]][j][8-i] = 1
                        # 对方
                        self.__red[105 + Chessman_2_idx[type(chess)]][9-j][i] = 1

        if self.t == 0:
            self.__red[112] = torch.FloatTensor(10, 9).fill_(1)
            self.__black[112] = torch.FloatTensor(10, 9).fill_(0)


    def update(self):
        self.t = self.t + 1
        self.make_features()


if __name__ == '__main__':
    cbd = chessboard.Chessboard('000')
    cbd.init_board()
    ob = Observation(cbd)
    print ob.red[105:112]
    ob.update()
    print ob.black[42:49]
