#-*- coding:utf-8 -*-

import chessman
from enum import Enum

Chessman_2_idx = {
    chessman.Pawn: 0,
    chessman.Cannon: 1,
    chessman.Rook: 2,
    chessman.Knight: 3,
    chessman.Elephant: 4,
    chessman.Mandarin: 5,
    chessman.King: 6
}

Idx_2_Chessman = {
    0: chessman.Pawn,
    1: chessman.Cannon,
    2: chessman.Rook,
    3: chessman.Knight,
    4: chessman.Elephant,
    5: chessman.Mandarin,
    6: chessman.King
}

class Color(Enum):
    Black = 0
    Red = 1
