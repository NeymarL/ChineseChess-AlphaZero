#-*- coding:utf-8 -*-

import chessman

class Chessboard(object):

    def __init__(self, name):
        self.__name = name
        self.__is_red_turn = True
        self.__chessmans = [([None] * 10) for i in range(9)]
        self.__chessmans_hash = {}
        self.turns = 1
        self.record = ''

    @property
    def is_red_turn(self):
        return self.__is_red_turn

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def chessmans(self):
        return self.__chessmans

    @property
    def chessmans_hash(self):
        return self.__chessmans_hash

    def init_board(self):
        red_rook_left = chessman.Rook(u" 车l红 ", "red_rook_left", True, self)
        red_rook_left.add_to_board(0, 0)
        red_rook_right = chessman.Rook(u" 车r红 ", "red_rook_right", True, self)
        red_rook_right.add_to_board(8, 0)
        black_rook_left = chessman.Rook(
            u" 车l黑 ", "black_rook_left", False, self)
        black_rook_left.add_to_board(0, 9)
        black_rook_right = chessman.Rook(
            u" 车r黑 ", "black_rook_right", False, self)
        black_rook_right.add_to_board(8, 9)
        red_knight_left = chessman.Knight(
            u" 马l红 ", "red_knight_left", True, self)
        red_knight_left.add_to_board(1, 0)
        red_knight_right = chessman.Knight(
            u" 马r红 ", "red_knight_right", True, self)
        red_knight_right.add_to_board(7, 0)
        black_knight_left = chessman.Knight(
            u" 马l黑 ", "black_knight_left", False, self)
        black_knight_left.add_to_board(1, 9)
        black_knight_right = chessman.Knight(
            u" 马r黑 ", "black_knight_right", False, self)
        black_knight_right.add_to_board(7, 9)
        red_cannon_left = chessman.Cannon(
            u" 炮l红 ", "red_cannon_left", True, self)
        red_cannon_left.add_to_board(1, 2)
        red_cannon_right = chessman.Cannon(
            u" 炮r红 ", "red_cannon_right", True, self)
        red_cannon_right.add_to_board(7, 2)
        black_cannon_left = chessman.Cannon(
            u" 炮l黑 ", "black_cannon_left", False, self)
        black_cannon_left.add_to_board(1, 7)
        black_cannon_right = chessman.Cannon(
            u" 炮r黑 ", "black_cannon_right", False, self)
        black_cannon_right.add_to_board(7, 7)
        red_elephant_left = chessman.Elephant(
            u" 相l红 ", "red_elephant_left", True, self)
        red_elephant_left.add_to_board(2, 0)
        red_elephant_right = chessman.Elephant(
            u" 相r红 ", "red_elephant_right", True, self)
        red_elephant_right.add_to_board(6, 0)
        black_elephant_left = chessman.Elephant(
            u" 象l黑 ", "black_elephant_left", False, self)
        black_elephant_left.add_to_board(2, 9)
        black_elephant_right = chessman.Elephant(
            u" 象r黑 ", "black_elephant_right", False, self)
        black_elephant_right.add_to_board(6, 9)
        red_mandarin_left = chessman.Mandarin(
            u" 仕l红 ", "red_mandarin_left", True, self)
        red_mandarin_left.add_to_board(3, 0)
        red_mandarin_right = chessman.Mandarin(
            u" 仕r红 ", "red_mandarin_right", True, self)
        red_mandarin_right.add_to_board(5, 0)
        black_mandarin_left = chessman.Mandarin(
            u" 仕l黑 ", "black_mandarin_left", False, self)
        black_mandarin_left.add_to_board(3, 9)
        black_mandarin_right = chessman.Mandarin(
            u" 仕r黑 ", "black_mandarin_right", False, self)
        black_mandarin_right.add_to_board(5, 9)
        red_king = chessman.King(u" 帅 红 ", "red_king", True, self)
        red_king.add_to_board(4, 0)
        black_king = chessman.King(u" 将 黑 ", "black_king", False, self)
        black_king.add_to_board(4, 9)
        red_pawn_1 = chessman.Pawn(u" 兵1红 ", "red_pawn_1", True, self)
        red_pawn_1.add_to_board(0, 3)
        red_pawn_2 = chessman.Pawn(u" 兵2红 ", "red_pawn_2", True, self)
        red_pawn_2.add_to_board(2, 3)
        red_pawn_3 = chessman.Pawn(u" 兵3红 ", "red_pawn_3", True, self)
        red_pawn_3.add_to_board(4, 3)
        red_pawn_4 = chessman.Pawn(u" 兵4红 ", "red_pawn_4", True, self)
        red_pawn_4.add_to_board(6, 3)
        red_pawn_5 = chessman.Pawn(u" 兵5红 ", "red_pawn_5", True, self)
        red_pawn_5.add_to_board(8, 3)
        black_pawn_1 = chessman.Pawn(u" 卒1黑 ", "black_pawn_1", False, self)
        black_pawn_1.add_to_board(0, 6)
        black_pawn_2 = chessman.Pawn(u" 卒2黑 ", "black_pawn_2", False, self)
        black_pawn_2.add_to_board(2, 6)
        black_pawn_3 = chessman.Pawn(u" 卒3黑 ", "black_pawn_3", False, self)
        black_pawn_3.add_to_board(4, 6)
        black_pawn_4 = chessman.Pawn(u" 卒4黑 ", "black_pawn_4", False, self)
        black_pawn_4.add_to_board(6, 6)
        black_pawn_5 = chessman.Pawn(u" 卒5黑 ", "black_pawn_5", False, self)
        black_pawn_5.add_to_board(8, 6)

    def add_chessman(self, chessman, col_num, row_num):
        self.chessmans[col_num][row_num] = chessman
        if chessman.name not in self.__chessmans_hash:
            self.__chessmans_hash[chessman.name] = chessman

    def remove_chessman_target(self, col_num, row_num):
        chessman_old = self.get_chessman(col_num, row_num)
        if chessman_old != None:
            self.__chessmans_hash.pop(chessman_old.name)
        return chessman_old

    def remove_chessman_source(self, col_num, row_num):
        self.chessmans[col_num][row_num] = None

    def calc_chessmans_moving_list(self):
        for chessman in self.__chessmans_hash.values():
            if chessman.is_red == self.__is_red_turn:
                chessman.calc_moving_list()

    def clear_chessmans_moving_list(self):
        for chessman in self.__chessmans_hash.values():
            chessman.clear_moving_list()

    def move_chessman(self, chessman, col_num, row_num, 
                      is_record = False, old_x = 0, old_y = 0):
        if chessman.is_red == self.__is_red_turn:
            chessman_old = self.remove_chessman_target(col_num, row_num)
            self.add_chessman(chessman, col_num, row_num)
            if self.is_check():
                if chessman_old != None:
                    self.add_chessman(chessman_old, col_num, row_num)
                else:
                    self.remove_chessman_source(col_num, row_num)
                return False
            if is_record:
                self.make_record(chessman, old_x, old_y, col_num, row_num)
            self.__is_red_turn = not self.__is_red_turn
            self.turns += self.__is_red_turn
            return True
        else:
            return False

    def is_end(self):
        if self.turns > 1000:
            print "Tie"
            return True
        if self.is_check():
            for i in range(9):
                for j in range(10):
                    chess = self.chessmans[i][j]
                    if chess != None and chess.is_red == self.is_red_turn:
                        for mov in chess.moving_list:
                            if chess.test_move(mov.x, mov.y):
                                print "+1s,", chess.name_cn, mov.x, mov.y
                                self.print_to_cl()
                                return False
        else:
            return False
        if self.is_red_turn:
            print "Black win"
        else:
            print "Red win"
        return True

    def get_chessman(self, col_num, row_num):
        return self.__chessmans[col_num][row_num]

    def get_chessman_by_name(self, name):
        if name in self.__chessmans_hash:
            return self.__chessmans_hash[name]

    def get_top_first_chessman(self, col_num, row_num):
        for i in range(row_num + 1, 10, 1):
            current = self.chessmans[col_num][i]
            if current != None:
                return current

    def get_bottom_first_chessman(self, col_num, row_num):
        for i in range(row_num - 1, -1, -1):
            current = self.chessmans[col_num][i]
            if current != None:
                return current

    def get_left_first_chessman(self, col_num, row_num):
        for i in range(col_num - 1, -1, -1):
            current = self.chessmans[i][row_num]
            if current != None:
                return current

    def get_right_first_chessman(self, col_num, row_num):
        for i in range(col_num + 1, 9, 1):
            current = self.chessmans[i][row_num]
            if current != None:
                return current

    def get_top_second_chessman(self, col_num, row_num):
        count = 0
        for i in range(row_num + 1, 10, 1):
            current = self.chessmans[col_num][i]
            if current != None:
                if count == 1:
                    return current
                else:
                    count += 1

    def get_bottom_second_chessman(self, col_num, row_num):
        count = 0
        for i in range(row_num - 1, -1, -1):
            current = self.chessmans[col_num][i]
            if current != None:
                if count == 1:
                    return current
                else:
                    count += 1

    def get_left_second_chessman(self, col_num, row_num):
        count = 0
        for i in range(col_num - 1, -1, -1):
            current = self.chessmans[i][row_num]
            if current != None:
                if count == 1:
                    return current
                else:
                    count += 1

    def get_right_second_chessman(self, col_num, row_num):
        count = 0
        for i in range(col_num + 1, 9, 1):
            current = self.chessmans[i][row_num]
            if current != None:
                if count == 1:
                    return current
                else:
                    count += 1

    def print_to_cl(self):
        screen = "\r\n"
        for i in range(9, -1, -1):
            for j in range(9):
                if self.__chessmans[j][i] != None:
                    screen += self.__chessmans[j][i].name_cn
                else:
                    screen += "   .   "
            screen += "\r\n" * 3
        print(screen)

    def is_check(self):
        global chessman
        if self.__is_red_turn:
            king = self.get_chessman_by_name("red_king")
        else:
            king = self.get_chessman_by_name("black_king")
        for i in range(9):
            for j in range(10):
                chess = self.chessmans[i][j]
                if chess != None:
                    chess.clear_moving_list()
                    chess.calc_moving_list()
                    if chess != None and chess.is_red != self.__is_red_turn:
                        if chess.in_moving_list(king.position.x, king.position.y):
                            # print "Checking:", chess.name, chess.position.x, chess.position.y
                            return True
        # the two king cannot exsits in one column with any obstacles
        red_king = self.get_chessman_by_name("red_king")
        black_king = self.get_chessman_by_name("black_king")
        checking = True
        if red_king.position.x == black_king.position.x:
            for i in range(red_king.position.y + 1, black_king.position.y):
                if self.chessmans[red_king.position.x][i] != None:
                    checking = False
        else:
            checking = False
        # if checking:
        #     print "Checking by kings", red_king.position.x, black_king.position.x
        return checking

    def check_position(self):
        for i in range(9):
            for j in range(10):
                chess = self.chessmans[i][j]
                if chess != None:
                    if chess.position.x != self.chessmans_hash[chess.name].position.x or\
                        chess.position.y != self.chessmans_hash[chess.name].position.y:
                        print "Error position:", chess.name, chess.position.x, chess.position.y

    def make_record(self, chess, old_x, old_y, x, y):
        if self.__is_red_turn:
            if self.turns != 1:
                self.record += '\n'
            self.record += str(self.turns) + '.'
        else:
            self.record += '\t'
        self.record += chess.name_cn[1]
        if old_y == y:
            if not self.is_red_turn:
                self.record += RECORD_NOTES[old_x + 1][0] + u'平' + RECORD_NOTES[x + 1][0]
            else:
                self.record += RECORD_NOTES[9 - old_x][1] + u'平' + RECORD_NOTES[9 - x][1]
        else:
            if not self.is_red_turn:
                self.record += RECORD_NOTES[old_x + 1][0]
            else:
                self.record += RECORD_NOTES[9 - old_x][1]
            if (y > old_y and self.is_red_turn) or (y < old_y and not self.is_red_turn):
                self.record += u'进'
            else:
                self.record += u'退'
            if type(chess) == chessman.Rook or type(chess) == chessman.Pawn or\
               type(chess) == chessman.Cannon or type(chess) == chessman.King:
               self.record += RECORD_NOTES[abs(y - old_y)][self.is_red_turn]
            else:
                if not self.is_red_turn:
                    self.record += RECORD_NOTES[x + 1][0]
                else:
                    self.record += RECORD_NOTES[9 - x][1]

    def print_record(self):
        print self.record

    def save_record(self, filename):
        pass

RECORD_NOTES = [
    ['0', '0'], ['1', u'一'], ['2', u'二'],
    ['3', u'三'], ['4', u'四'], ['5', u'五'],
    ['6', u'六'], ['7', u'七'], ['8', u'八'],
    ['9', u'九']
]

