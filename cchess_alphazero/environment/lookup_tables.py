#-*- coding:utf-8 -*-

from cchess_alphazero.environment.chessman import *
from enum import Enum

Chessman_2_idx = {
    Pawn: 0,
    Cannon: 1,
    Rook: 2,
    Knight: 3,
    Elephant: 4,
    Mandarin: 5,
    King: 6
}

Idx_2_Chessman = {
    0: Pawn,
    1: Cannon,
    2: Rook,
    3: Knight,
    4: Elephant,
    5: Mandarin,
    6: King
}

Fen_2_Idx = {
    'p': 0,
    'P': 0,
    'c': 1,
    'C': 1,
    'r': 2,
    'R': 2,
    'k': 3,
    'K': 3,
    'e': 4,
    'E': 4,
    'm': 5,
    'M': 5,
    's': 6,
    'S': 6
}

class Color(Enum):
    Black = 0
    Red = 1

Winner = Enum("Winner", "red black draw")

