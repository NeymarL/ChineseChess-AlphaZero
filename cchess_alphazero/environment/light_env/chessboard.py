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
from cchess_alphazero.environment.lookup_tables import Winner

class L_Chessboard:

    def __init__(self):
        self.height = 10
        self.width = 9
        self.board = [['.' for col in range(self.width)] for row in range(self.height)]
        self.steps = 0
        self._legal_moves = None
        self.turn = RED
        self.assign_fen(None)
        self.winner = None

    def _update(self):
        self._fen = None
        self._legal_moves = None
        self.steps += 1
        if self.steps % 2 == 0:
            self.turn = RED
        else:
            self.turn = BLACK

    def assign_fen(self, fen):
        if fen is None:
            fen = init_fen
        x = 0
        y = 0
        for k in range(0, len(fen)):
            ch = fen[k]
            if ch == ' ':
                if (fen[k+1] == 'b'):
                    self.turn = BLACK
                break
            if ch == '/':
                x = 0
                y += 1
            elif ch >= '1' and ch <= '9':
                for i in range(int(ch)):
                    self.board[y][x] = '.'
                    x = x+1
            else:
                self.board[y][x] = ch
                x = x+1

    def FENboard(self):
        def swapcase(a):
            if a.isalpha():
                a = replace_dict[a]
                return a.lower() if a.isupper() else a.upper()
            return a

        c = 0
        fen = ''
        for i in range(self.height - 1, -1, -1):
            c = 0
            for j in range(self.width):
                if self.board[i][j] == '.':
                    c = c + 1
                else:
                    if c > 0:
                        fen = fen + str(c)
                    fen = fen + swapcase(self.board[i][j])
                    c = 0
            if c > 0:
                fen = fen + str(c)
            if i > 0:
                fen = fen + '/'
        if self.turn is RED:
            fen += ' r'
        else:
            fen += ' b'
        fen += ' - - 0 1'
        return fen

    def fliped_FENboard(self):
        fen = self.FENboard()
        foo = fen.split(' ')
        rows = foo[0].split('/')
        def swapcase(a):
            if a.isalpha():
                return a.lower() if a.isupper() else a.upper()
            return a
        def swapall(aa):
            return "".join([swapcase(a) for a in aa])

        return "/".join([swapall(reversed(row)) for row in reversed(rows)]) \
            + " " + foo[1] \
            + " " + foo[2] \
            + " " + foo[3] + " " + foo[4] + " " + foo[5]

    @property
    def is_red_turn(self):
        return self.turn == RED

    @property
    def screen(self):
        return self.board

    def legal_moves(self):
        if self._legal_moves is not None:
            return self._legal_moves

        _legal_moves = []
        for y in range(self.height):
            for x in range(self.width):
                ch = self.board[y][x]
                if (self.turn == RED and ch.isupper()):
                    continue
                if (self.turn == BLACK and ch.islower()):
                    continue
                if ch in mov_dir:
                    if (x == 0 and y == 3):
                        aa = 3
                    for d in mov_dir[ch]:
                        x_ = x + d[0]
                        y_ = y + d[1]
                        if not self._can_move(x_, y_):
                            continue
                        elif ch == 'p' and y < 5 and x_ != x:  # for red pawn
                            continue
                        elif ch == 'P' and y > 4 and x_ != x:  # for black pawn
                            continue
                        elif ch == 'n' or ch == 'N' or ch == 'b' or ch == 'B': # for knight and bishop
                            if self.board[y+int(d[1]/2)][x+int(d[0]/2)] != '.':
                                continue
                            elif ch == 'b' and y_ > 4:
                                continue
                            elif ch == 'B' and y_ < 5:
                                continue
                        elif ch != 'p' and ch != 'P': # for king and advisor
                            if x_ < 3 or x_ > 5:
                                continue
                            if (ch == 'k' or ch == 'a') and y_ > 2:
                                continue
                            if (ch == 'K' or ch == 'A') and y_ < 7:
                                continue
                        _legal_moves.append(move_to_str(x, y, x_, y_))
                        if (ch == 'k' and self.turn == RED): #for King to King check
                            d, u = self._y_board_from(x, y)
                            if (u < self.height and self.board[u][x] == 'K'):
                                _legal_moves.append(move_to_str(x, y, x, u))
                        elif (ch == 'K' and self.turn == BLACK):
                            d, u = self._y_board_from(x, y)
                            if (d > -1 and self.board[d][x] == 'k'):
                                _legal_moves.append(move_to_str(x, y, x, d))
                elif ch != '.': # for connon and root
                    l,r = self._x_board_from(x,y)
                    d,u = self._y_board_from(x,y)
                    for x_ in range(l+1,x):
                        _legal_moves.append(move_to_str(x, y, x_, y))
                    for x_ in range(x+1,r):
                        _legal_moves.append(move_to_str(x, y, x_, y))
                    for y_ in range(d+1,y):
                        _legal_moves.append(move_to_str(x, y, x, y_))
                    for y_ in range(y+1,u):
                        _legal_moves.append(move_to_str(x, y, x, y_))
                    if ch == 'r' or ch == 'R': # for root
                        if self._can_move(l, y):
                            _legal_moves.append(move_to_str(x, y, l, y))
                        if self._can_move(r, y):
                            _legal_moves.append(move_to_str(x, y, r, y))
                        if self._can_move(x, d):
                            _legal_moves.append(move_to_str(x, y, x, d))
                        if self._can_move(x, u):
                            _legal_moves.append(move_to_str(x, y, x, u))
                    else: # for connon
                        l_, _ = self._x_board_from(l,y)
                        _, r_ = self._x_board_from(r,y)
                        d_, _ = self._y_board_from(x,d)
                        _, u_ = self._y_board_from(x,u)
                        if self._can_move(l_, y):
                            _legal_moves.append(move_to_str(x, y, l_, y))
                        if self._can_move(r_, y):
                            _legal_moves.append(move_to_str(x, y, r_, y))
                        if self._can_move(x, d_):
                            _legal_moves.append(move_to_str(x, y, x, d_))
                        if self._can_move(x, u_):
                            _legal_moves.append(move_to_str(x, y, x, u_))

        self._legal_moves = _legal_moves
        return _legal_moves

    def is_legal(self, mov):
        return mov.uci in self.legal_moves

    def is_end(self):
        red_k, black_k = [0, 0], [0, 0]
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i][j] == 'k':
                    red_k[0] = i
                    red_k[1] = j
                if self.board[i][j] == 'K':
                    black_k[0] = i
                    black_k[1] = j
        if red_k[0] == 0 and red_k[1] == 0:
            self.winner = Winner.black
        elif black_k[0] == 0 and black_k[1] == 0:
            self.winner = Winner.red
        elif red_k[1] == black_k[1]:
            has_block = False
            i = red_k[0] + 1
            while i < black_k[0]:
                if self.board[i][red_k[1]] != '.':
                    has_block = True
                    break
                i += 1
            if not has_block:
                if self.turn == RED:
                    self.winner = Winner.red
                else:
                    self.winner = Winner.black
        return self.winner is not None

    def print_to_cl(self):
        print(self.board)

    def move_action_str(self, uci):
        mov = Move(uci)
        self.push(mov)
        return True

    def push(self, mov):
        self.board[mov.n[1]][mov.n[0]] = self.board[mov.p[1]][mov.p[0]]
        self.board[mov.p[1]][mov.p[0]] = '.'
        self._update()


    def _is_same_side(self,x,y):
        if self.turn == RED and self.board[y][x].islower():
            return True
        if self.turn == BLACK and self.board[y][x].isupper():
            return True

    def _can_move(self,x,y): # basically check the move
        if x < 0 or x > self.width-1:
            return False
        if y < 0 or y > self.height-1:
            return False
        if self._is_same_side(x,y):
            return False
        return True

    def _x_board_from(self,x,y):
        l = x-1
        r = x+1
        while l > -1 and self.board[y][l] == '.':
            l = l-1
        while r < self.width and self.board[y][r] == '.':
            r = r+1
        return l,r

    def _y_board_from(self,x,y):
        d = y-1
        u = y+1
        while d > -1 and self.board[d][x] == '.':
            d = d-1
        while u < self.height and self.board[u][x] == '.':
            u = u+1
        return d,u

    def result(self, claim_draw=True) -> str:
        rst = '*'
        if ('k' not in self.board[0]) and ('k' not in self.board[1]) and ('k' not in self.board[2]):
            rst = '0-1'
        if ('K' not in self.board[9]) and ('K' not in self.board[8]) and ('K' not in self.board[7]):
            rst = '1-0'
        return rst

    def clear_chessmans_moving_list(self):
        return

    def calc_chessmans_moving_list(self):
        return

    def save_record(self, filename):
        return

    def parse_WXF_move(self, wxf):
        '''
        red is upper, black is lower alphabet
        '''
        p = self.swapcase(wxf[0])
        col = wxf[1]
        mov = wxf[2]
        dest_col = wxf[3]
        src_row, src_col = self.find_row(p, col)
        if mov == '.' or mov == '=':
            # move horizontally
            dest_row = src_row
            if p.islower():
                dest_col = int(dest_col) - 1
            else:
                dest_col = self.width - int(dest_col)
        else:
            if p == 'h' or p == 'H' or p == 'e' or p == 'E' or p == 'a' or p == 'A':
                if p.islower():
                    dest_col = int(dest_col) - 1
                else:
                    dest_col = self.width - int(dest_col)

                if p == 'h' or p == 'H':
                    # for house/knight
                    step = 1 if abs(dest_col - src_col) == 2 else 2
                elif p == 'e' or p == 'E':
                    # for elephant/bishop
                    step = 2
                else:
                    # for advisor
                    step = 1 
                if mov == '+' and p.islower() or mov == '-' and p.isupper():
                    dest_row = src_row + step
                else:
                    dest_row = src_row - step
            else:
                # move vertically
                step = int(dest_col)
                if mov == '+' and p.islower() or mov == '-' and p.isupper():
                    dest_row = src_row + step
                else:
                    dest_row = src_row - step
                dest_col = src_col
        return move_to_str(src_col, src_row, dest_col, dest_row)

    def find_row(self, piece, col):
        if piece == 'h' or piece == 'H':
            piece = 'n' if piece == 'h' else 'N'
        if piece == 'e' or piece == 'E':
            piece = 'b' if piece == 'e' else 'B'
        column = 0
        row = -1
        if col.isdigit():
            if piece.isupper():
                column = self.width - int(col)
            else:
                column = int(col) - 1
            for i in range(self.height):
                if self.board[i][int(column)] == piece:
                    row = i
                    break
        else:
            first_row = -1
            second_row = -1
            column = -1
            for j in range(self.width):
                column = -1
                for i in range(self.height):
                    if self.board[i][j] == piece:
                        if column == -1:
                            column = j
                            first_row = i
                        else:
                            if column == j:
                                second_row = i
                                break
                            else:
                                column = j
                                first_row = second_row = -1
                if first_row != -1 and second_row != -1:
                    break
            if (piece.islower() and col == '+') or (piece.isupper() and col == '-'):
                row = second_row
            else:
                row = first_row
        return row, column

    def swapcase(self, a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a



if __name__ == '__main__': # test
    board = Chessboard()
    print(board.legal_moves)