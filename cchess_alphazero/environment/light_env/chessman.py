#! /usr/bin/env python
# -*- coding: utf-8 -*-

# pycchess - just another chinese chess UI
# Copyright (C) 2011 - 2015 timebug

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from cchess_alphazero.environment.light_env.common import *


class L_Chessman:
    def __init__(self, kind, color, x, y, pc):
        self.kind = kind
        self.color = color
        self.x = x
        self.y = y
        self.pc = pc
        self.over_river = False

    def move_check(self, x, y): # within the feasible region
        ok = True
        if x < 0 or x > 8:
            return False
        if y < 0 or y > 9:
            return False
        if self.kind == KING:
            if abs(self.x - x) + abs(self.y - y) != 1:
                ok = False
            if x < 3 or x > 5:
                ok = False
            if self.y < 3 and y >= 3:
                ok = False
            if self.y > 6 and y <= 6:
                ok = False
        elif self.kind == ADVISOR:
            if abs(self.x - x) != 1 or abs(self.y - y) != 1:
                ok = False
            if x < 3 or x > 5:
                ok = False
            if self.y < 3 and y >= 3:
                ok = False
            if self.y > 6 and y <= 6:
                ok = False
        elif self.kind == BISHOP:
            if abs(self.x - x) != 2 or abs(self.y - y) != 2:
                ok = False
            if self.y < 5 and y >= 5:
                ok = False
            if self.y > 4 and y <= 4:
                ok = False
        elif self.kind == KNIGHT:
            if pow((self.x - x), 2) + pow((self.y - y), 2) != 5:
                ok = False
        elif self.kind == ROOK or self.kind == CANNON:
            if abs(self.x - x) != 0 and abs(self.y - y) != 0:
                ok = False
        elif self.kind == PAWN:
            if self.over_river:
                if abs(self.x - x) + abs(self.y - y) != 1:
                    ok = False
                if self.y < 5:
                    if self.x == x and self.y - y != 1:
                        ok = False
                else:
                    if self.x == x and y - self.y != 1:
                        ok = False
            else:
                if self.y < 5:
                    if y - self.y != 1 or x != self.x:
                        ok = False
                    else:
                        if self.y == 4 and y == 5:
                            self.over_river = True
                else:
                    if self.y - y != 1 or x != self.x:
                        ok = False
                    else:
                        if self.y == 5 and y == 4:
                            self.over_river = True

        else:
            ok = False
        return ok
