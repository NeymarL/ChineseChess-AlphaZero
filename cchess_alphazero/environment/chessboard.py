import codecs

from cchess_alphazero.environment.lookup_tables import Winner
from cchess_alphazero.environment.chessman import *

from cchess_alphazero.lib.logger import getLogger

logger = getLogger(__name__)

class Chessboard(object):

    def __init__(self, name='000'):
        self.__name = name
        self.__is_red_turn = True
        self.__chessmans = [([None] * 10) for i in range(9)]
        self.__chessmans_hash = {}
        self.turns = 1
        self.record = ''
        self.winner = None
        self.__screen = ''

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

    @property
    def screen(self):
        self.print_to_cl(is_print=False)
        return self.__screen

    def init_board(self):
        red_rook_left = Rook(u" 车l红 ", "red_rook_left", True, self, 'R')
        red_rook_left.add_to_board(0, 0)
        red_rook_right = Rook(u" 车r红 ", "red_rook_right", True, self, 'R')
        red_rook_right.add_to_board(8, 0)
        black_rook_left = Rook(
            u" 车l黑 ", "black_rook_left", False, self, 'r')
        black_rook_left.add_to_board(0, 9)
        black_rook_right = Rook(
            u" 车r黑 ", "black_rook_right", False, self, 'r')
        black_rook_right.add_to_board(8, 9)
        red_knight_left = Knight(
            u" 马l红 ", "red_knight_left", True, self, 'K')
        red_knight_left.add_to_board(1, 0)
        red_knight_right = Knight(
            u" 马r红 ", "red_knight_right", True, self, 'K')
        red_knight_right.add_to_board(7, 0)
        black_knight_left = Knight(
            u" 马l黑 ", "black_knight_left", False, self, 'k')
        black_knight_left.add_to_board(1, 9)
        black_knight_right = Knight(
            u" 马r黑 ", "black_knight_right", False, self, 'k')
        black_knight_right.add_to_board(7, 9)
        red_cannon_left = Cannon(
            u" 炮l红 ", "red_cannon_left", True, self, 'C')
        red_cannon_left.add_to_board(1, 2)
        red_cannon_right = Cannon(
            u" 炮r红 ", "red_cannon_right", True, self, 'C')
        red_cannon_right.add_to_board(7, 2)
        black_cannon_left = Cannon(
            u" 炮l黑 ", "black_cannon_left", False, self, 'c')
        black_cannon_left.add_to_board(1, 7)
        black_cannon_right = Cannon(
            u" 炮r黑 ", "black_cannon_right", False, self, 'c')
        black_cannon_right.add_to_board(7, 7)
        red_elephant_left = Elephant(
            u" 相l红 ", "red_elephant_left", True, self, 'E')
        red_elephant_left.add_to_board(2, 0)
        red_elephant_right = Elephant(
            u" 相r红 ", "red_elephant_right", True, self, 'E')
        red_elephant_right.add_to_board(6, 0)
        black_elephant_left = Elephant(
            u" 象l黑 ", "black_elephant_left", False, self, 'e')
        black_elephant_left.add_to_board(2, 9)
        black_elephant_right = Elephant(
            u" 象r黑 ", "black_elephant_right", False, self, 'e')
        black_elephant_right.add_to_board(6, 9)
        red_mandarin_left = Mandarin(
            u" 仕l红 ", "red_mandarin_left", True, self, 'M')
        red_mandarin_left.add_to_board(3, 0)
        red_mandarin_right = Mandarin(
            u" 仕r红 ", "red_mandarin_right", True, self, 'M')
        red_mandarin_right.add_to_board(5, 0)
        black_mandarin_left = Mandarin(
            u" 仕l黑 ", "black_mandarin_left", False, self, 'm')
        black_mandarin_left.add_to_board(3, 9)
        black_mandarin_right = Mandarin(
            u" 仕r黑 ", "black_mandarin_right", False, self, 'm')
        black_mandarin_right.add_to_board(5, 9)
        red_king = King(u" 帅 红 ", "red_king", True, self, 'S')
        red_king.add_to_board(4, 0)
        black_king = King(u" 将 黑 ", "black_king", False, self, 's')
        black_king.add_to_board(4, 9)
        red_pawn_1 = Pawn(u" 兵1红 ", "red_pawn_1", True, self, 'P')
        red_pawn_1.add_to_board(0, 3)
        red_pawn_2 = Pawn(u" 兵2红 ", "red_pawn_2", True, self, 'P')
        red_pawn_2.add_to_board(2, 3)
        red_pawn_3 = Pawn(u" 兵3红 ", "red_pawn_3", True, self, 'P')
        red_pawn_3.add_to_board(4, 3)
        red_pawn_4 = Pawn(u" 兵4红 ", "red_pawn_4", True, self, 'P')
        red_pawn_4.add_to_board(6, 3)
        red_pawn_5 = Pawn(u" 兵5红 ", "red_pawn_5", True, self, 'P')
        red_pawn_5.add_to_board(8, 3)
        black_pawn_1 = Pawn(u" 卒1黑 ", "black_pawn_1", False, self, 'p')
        black_pawn_1.add_to_board(0, 6)
        black_pawn_2 = Pawn(u" 卒2黑 ", "black_pawn_2", False, self, 'p')
        black_pawn_2.add_to_board(2, 6)
        black_pawn_3 = Pawn(u" 卒3黑 ", "black_pawn_3", False, self, 'p')
        black_pawn_3.add_to_board(4, 6)
        black_pawn_4 = Pawn(u" 卒4黑 ", "black_pawn_4", False, self, 'p')
        black_pawn_4.add_to_board(6, 6)
        black_pawn_5 = Pawn(u" 卒5黑 ", "black_pawn_5", False, self, 'p')
        black_pawn_5.add_to_board(8, 6)
        self.calc_chessmans_moving_list()

    def add_chessman(self, chessman, col_num, row_num):
        self.chessmans[col_num][row_num] = chessman
        if chessman.name not in self.__chessmans_hash:
            self.__chessmans_hash[chessman.name] = chessman

    def remove_chessman_target(self, col_num, row_num):
        chessman_old = self.get_chessman(col_num, row_num)
        if chessman_old != None:
            self.__chessmans_hash.pop(chessman_old.name)
            chessman_old.is_alive = False
        return chessman_old

    def remove_chessman_source(self, col_num, row_num):
        self.chessmans[col_num][row_num] = None

    def calc_chessmans_moving_list(self):
        for chessman in self.__chessmans_hash.values():
            if chessman.is_red == self.__is_red_turn:
                chessman.clear_moving_list()
                chessman.calc_moving_list()

    def clear_chessmans_moving_list(self):
        for chessman in self.__chessmans_hash.values():
            chessman.clear_moving_list()

    def move_chessman(self, chessman, col_num, row_num, 
                      is_record = False, old_x = 0, old_y = 0):
        if chessman.is_red == self.__is_red_turn:
            chessman_old = self.remove_chessman_target(col_num, row_num)
            self.add_chessman(chessman, col_num, row_num)
            # if self.is_check():
            #     if chessman_old != None:
            #         self.add_chessman(chessman_old, col_num, row_num)
            #     else:
            #         self.remove_chessman_source(col_num, row_num)
            #     return False
            if is_record:
                self.make_record(chessman, old_x, old_y, col_num, row_num)
            self.__is_red_turn = not self.__is_red_turn
            self.turns += self.__is_red_turn
            return True
        else:
            return False

    def move(self, x0, y0, x1, y1):
        chessman = self.chessmans[x0][y0]
        if chessman == None:
            return False
        return chessman.move(x1, y1)

    def move_action_str(self, action):
        x0, y0, x1, y1 = self.str_to_move(action)
        return self.move(x0, y0, x1, y1)

    def legal_moves(self):
        '''
        return all legal moves
        '''
        _legal_moves = []
        for chessman in self.__chessmans_hash.values():
            if chessman.is_red == self.is_red_turn:
                p = chessman.position
                x0 = p.x
                y0 = p.y
                print(f"{chessman.name_cn}, pos ({x0}, {y0})")
                for point in chessman.moving_list:
                    _legal_moves.append(self.move_to_str(x0, y0, point.x, point.y))
        return _legal_moves


    def is_end(self):
        red_king = self.get_chessman_by_name('red_king')
        black_king = self.get_chessman_by_name('black_king')
        if not red_king:
            self.winner = Winner.black
        elif not black_king:
            self.winner = Winner.red
        elif red_king.position.x == black_king.position.x:
            checking = True
            for i in range(red_king.position.y + 1, black_king.position.y):
                if self.chessmans[red_king.position.x][i] != None:
                    checking = False
                    break
            if checking:
                if self.is_red_turn:
                    self.winner = Winner.red
                else:
                    self.winner = Winner.black
        return self.winner != None


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

    def print_to_cl(self, is_print = True):
        screen = "\r\n"
        for i in range(9, -1, -1):
            for j in range(9):
                if self.__chessmans[j][i] != None:
                    screen += self.__chessmans[j][i].name_cn
                else:
                    screen += "   .   "
            screen += "\r\n" * 3
        if is_print:
            print(screen)
        else:
            self.__screen = screen

    def is_check(self):
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
                            # logger.debug("Checking:", chess.name, chess.position.x, chess.position.y)
                            return True
        # the two king cannot exsits in one column without any obstacles
        red_king = self.get_chessman_by_name("red_king")
        black_king = self.get_chessman_by_name("black_king")
        checking = True
        if red_king.position.x == black_king.position.x:
            for i in range(red_king.position.y + 1, black_king.position.y):
                if self.chessmans[red_king.position.x][i] != None:
                    checking = False
        else:
            checking = False
        return checking

    def check_position(self):
        for i in range(9):
            for j in range(10):
                chess = self.chessmans[i][j]
                if chess != None:
                    if chess.position.x != self.chessmans_hash[chess.name].position.x or\
                        chess.position.y != self.chessmans_hash[chess.name].position.y:
                        print("Error position:", chess.name, chess.position.x, chess.position.y)

    def make_record(self, chess, old_x, old_y, x, y):
        if self.__is_red_turn:
            if self.turns != 1:
                self.record += '\n'
            self.record += str(self.turns) + '.'
        else:
            self.record += '\t'
        has_two, mark = self.check_two_chesses_in_one_row(chess, old_x, old_y)
        if has_two:
            self.record += mark
        self.record += chess.name_cn[1]
        # horizontal move
        if old_y == y:
            if not self.is_red_turn:
                if not has_two:
                    self.record += RECORD_NOTES[old_x + 1][0]
                self.record += u'平' + RECORD_NOTES[x + 1][0]
            else:
                if not has_two:
                    self.record += RECORD_NOTES[9 - old_x][1]
                self.record += u'平' + RECORD_NOTES[9 - x][1]
        # vertical move
        else:
            if not has_two:
                if not self.is_red_turn:
                    self.record += RECORD_NOTES[old_x + 1][0]
                else:
                    self.record += RECORD_NOTES[9 - old_x][1]
            if (y > old_y and self.is_red_turn) or (y < old_y and not self.is_red_turn):
                self.record += u'进'
            else:
                self.record += u'退'
            if type(chess) == Rook or type(chess) == Pawn or\
               type(chess) == Cannon or type(chess) == King:
               self.record += RECORD_NOTES[abs(y - old_y)][self.is_red_turn]
            else:
                if not self.is_red_turn:
                    self.record += RECORD_NOTES[x + 1][0]
                else:
                    self.record += RECORD_NOTES[9 - x][1]

    def check_two_chesses_in_one_row(self, chess, old_x, old_y):
        for j in range(10):
            chs = self.chessmans[old_x][j]
            if chs != None and chs.is_red == chess.is_red:
                if type(chs) == type(chess) and chs != chess:
                    if (chs.position.y > old_y and not chs.is_red) or\
                       (chs.position.y < old_y and chs.is_red):
                        return (True, u'前')
                    else:
                        return (True, u'后')
        return (False, u'')

    def print_record(self):
        print(self.record)

    def save_record(self, filename, head = ''):
        with codecs.open(filename, "a", encoding="utf-8") as f:
            if head != '':
                f.write(head)
            f.write(self.record)

    def str_to_move(self, action: str):
        x0 = int(action[0])
        y0 = int(action[1])
        x1 = int(action[2])
        y1 = int(action[3])
        return x0, y0, x1, y1

    def move_to_str(self, x0, y0, x1, y1):
        return str(x0) + str(y0) + str(x1) + str(y1)

    def FENboard(self):
        '''
        FEN board representation
        rules: https://www.xqbase.com/protocol/pgnfen2.htm
        '''
        cnt = 0
        fen = ''
        for i in range(9, -1, -1):
            cnt = 0
            for j in range(9):
                if self.chessmans[j][i] == None:
                    cnt += 1
                else:
                    if cnt > 0:
                        fen = fen + str(cnt)
                    fen = fen + self.chessmans[j][i].fen
                    cnt = 0
            if cnt > 0:
                fen = fen + str(cnt)
            if i > 0:
                fen = fen + '/'
        fen += ' r'
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
            + " " + ('r' if foo[1] == 'b' else 'b') \
            + " " + foo[2] \
            + " " + foo[3] + " " + foo[4] + " " + foo[5]

    def make_single_record(self, old_x, old_y, x, y):
        record = ''
        if not self.is_red_turn:
            old_y = 9 - old_y
            y = 9 - y
            x = 8 - x
            old_x = 8 - old_x
        chess = self.chessmans[old_x][old_y]
        if chess is None:
            logger.error(f"No chessman! {old_x}{old_y}{x}{y}")
            self.print_to_cl()
        has_two, mark = self.check_two_chesses_in_one_row(chess, old_x, old_y)
        if has_two:
            record += mark
        record += chess.name_cn[1]
        # horizontal move
        if old_y == y:
            if not self.is_red_turn:
                if not has_two:
                    record += RECORD_NOTES[old_x + 1][0]
                record += u'平' + RECORD_NOTES[x + 1][0]
            else:
                if not has_two:
                    record += RECORD_NOTES[9 - old_x][1]
                record += u'平' + RECORD_NOTES[9 - x][1]
        # vertical move
        else:
            if not has_two:
                if not self.is_red_turn:
                    record += RECORD_NOTES[old_x + 1][0]
                else:
                    record += RECORD_NOTES[9 - old_x][1]
            if (y > old_y and self.is_red_turn) or (y < old_y and not self.is_red_turn):
                record += u'进'
            else:
                record += u'退'
            if type(chess) == Rook or type(chess) == Pawn or\
               type(chess) == Cannon or type(chess) == King:
               record += RECORD_NOTES[abs(y - old_y)][self.is_red_turn]
            else:
                if not self.is_red_turn:
                    record += RECORD_NOTES[x + 1][0]
                else:
                    record += RECORD_NOTES[9 - x][1]
        return record




RECORD_NOTES = [
    ['0', '0'], ['1', u'一'], ['2', u'二'],
    ['3', u'三'], ['4', u'四'], ['5', u'五'],
    ['6', u'六'], ['7', u'七'], ['8', u'八'],
    ['9', u'九']
]

