import copy

class Point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y

def num_between(max_num, min_num, current):
    return current >= min_num and current <= max_num


def creat_points(list_points, list_vs, list_hs):
    for v in list_vs:
        for h in list_hs:
            list_points.append(Point(v, h))


class Chessman(object):

    def __init__(self, name_cn, name, is_red, chessboard, fen):
        self.__name = name
        self.__is_red = is_red
        self.__chessboard = chessboard
        self.__position = Point(None, None)
        self.__moving_list = []
        self.__top = 9
        self.__bottom = 0
        self.__left = 0
        self.__right = 8
        self.__is_alive = True
        self.__name_cn = name_cn
        self.__fen = fen

    @property
    def row_num(self):
        return self.__position.y

    @property
    def col_num(self):
        return self.__position.x

    @property
    def is_alive(self):
        return self.__is_alive

    @is_alive.setter
    def is_alive(self, is_alive):
        self.__is_alive = is_alive

    @property
    def chessboard(self):
        return self.__chessboard

    @property
    def is_red(self):
        return self.__is_red

    @property
    def name(self):
        return self.__name

    @property
    def name_cn(self):
        return self.__name_cn

    @property
    def position(self):
        return self.__position

    @property
    def moving_list(self):
        return self.__moving_list

    @property
    def fen(self):
        return self.__fen

    def reset_board(self, chessboard):
        self.__chessboard = chessboard

    def clear_moving_list(self):
        self.__moving_list = []

    def add_to_board(self, col_num, row_num):
        if self.border_check(col_num, row_num):
            self.__position.x = col_num
            self.__position.y = row_num
            self.__chessboard.add_chessman(self, col_num, row_num)
        else:
            print("the worng postion")

    def move(self, col_num, row_num):
        if self.in_moving_list(col_num, row_num):
            self.__chessboard.remove_chessman_source(self.__position.x, self.__position.y)
            old_x = self.__position.x
            old_y = self.__position.y
            self.__position.x = col_num
            self.__position.y = row_num
            if not self.__chessboard.move_chessman(self, col_num, row_num, True, old_x, old_y):
                self.__position.x = old_x
                self.__position.y = old_y
                self.__chessboard.add_chessman(self, self.__position.x, self.__position.y)
                self.clear_moving_list()
                self.calc_moving_list()
                # self.__chessboard.print_to_cl()
                return False
            return True
        else:
            self.clear_moving_list()
            self.calc_moving_list()
            if self.in_moving_list(col_num, row_num):
                return self.move(col_num, row_num)
            print("the worng target_position:", self.name_cn, col_num, row_num)
            for point in self.moving_list:
                print(point.x, point.y)
            return False

    def test_move(self, col_num, row_num):
        if self.in_moving_list(col_num, row_num):
            chessman = copy.deepcopy(self)
            chessboard = copy.deepcopy(self.__chessboard)
            chessman.reset_board(chessboard)
            return chessman.move(col_num, row_num)
        else:
            return False
        # if self.in_moving_list(col_num, row_num):
        #     self.__chessboard.remove_chessman_source(self.__position.x, self.__position.y)
        #     old_x = self.__position.x
        #     old_y = self.__position.y
        #     self.__position.x = col_num
        #     self.__position.y = row_num
        #     chessman_old = self.chessboard.remove_chessman_target(col_num, row_num)
        #     self.chessboard.add_chessman(self, col_num, row_num)
        #     # is check
        #     checking = self.chessboard.is_check()
        #     # restore
        #     if chessman_old != None:
        #         self.chessboard.add_chessman(chessman_old, col_num, row_num)
        #     else:
        #         self.chessboard.remove_chessman_source(col_num, row_num)
        #     self.__position.x = old_x
        #     self.__position.y = old_y
        #     self.__chessboard.add_chessman(self, self.__position.x, self.__position.y)
        #     self.clear_moving_list()
        #     self.calc_moving_list()
        #     return not checking
        # else:
        #     return False

    def in_moving_list(self, col_num, row_num):
        for point in self.__moving_list:
            if point.x == col_num and point.y == row_num:
                return True
        return False

    def calc_moving_list(self):
        pass

    def border_check(self, col_num, row_num):
        return num_between(self.__top, self.__bottom, row_num) and num_between(self.__right, self.__left, col_num)

    def calc_moving_path(self, direction_chessman, direction_vertical_coordinate, 
                        current_vertical_coordinate, direction_parallel_coordinate, direction, 
                        border_vertical_coordinate, h_or_v, ignore_color=False):
        if direction_chessman != None:
            if direction_chessman.is_red == self.is_red or ignore_color:
                for i in range(direction_vertical_coordinate + direction, current_vertical_coordinate, direction):
                    self.__moving_list.append(
                        Point(i, direction_parallel_coordinate) if h_or_v else Point(direction_parallel_coordinate, i))

            else:
                for i in range(direction_vertical_coordinate, current_vertical_coordinate, direction):
                    self.__moving_list.append(
                        Point(i, direction_parallel_coordinate) if h_or_v else Point(direction_parallel_coordinate, i))
        else:
            for i in range(border_vertical_coordinate, current_vertical_coordinate, direction):
                self.__moving_list.append(
                    Point(i, direction_parallel_coordinate) if h_or_v else Point(direction_parallel_coordinate, i))

    def add_from_probable_points(self, probable_moving_points, current_color):
        for point in probable_moving_points:
            if self.border_check(point.x, point.y):
                chessman = self.chessboard.get_chessman(
                    point.x, point.y)
                if chessman is None or chessman.is_red != current_color:
                    self.moving_list.append(point)


class Rook(Chessman):
    '''车'''

    def __init__(self, name_cn, name, is_red, chessboard, fen):
        super(Rook, self).__init__(name_cn, name, is_red, chessboard, fen)
        self._Chessman__top = 9
        self._Chessman__bottom = 0
        self._Chessman__left = 0
        self._Chessman__right = 8

    def calc_moving_list(self):
        current_v_c = super(Rook, self).position.x
        current_h_c = super(Rook, self).position.y
        left = super(Rook, self).chessboard.get_left_first_chessman(
            current_v_c, current_h_c)
        right = super(Rook, self).chessboard.get_right_first_chessman(
            current_v_c, current_h_c)
        top = super(Rook, self).chessboard.get_top_first_chessman(
            current_v_c, current_h_c)
        bottom = super(Rook, self).chessboard.get_bottom_first_chessman(
            current_v_c, current_h_c)

        super(Rook, self).calc_moving_path(left, (left.position.x if left != None else None),
                                           current_v_c, current_h_c, 1, 0, True)
        super(Rook, self).calc_moving_path(right, (right.position.x if right != None else None),
                                           current_v_c, current_h_c, -1, 8, True)
        super(Rook, self).calc_moving_path(top, (top.position.y if top != None else None),
                                           current_h_c, current_v_c, -1, 9, False)
        super(Rook, self).calc_moving_path(bottom, (bottom.position.y if bottom != None else None),
                                           current_h_c, current_v_c, 1, 0, False)


class Knight(Chessman):
    '''马'''

    def __init__(self, name_cn, name, is_red, chessboard, fen):
        super(Knight, self).__init__(name_cn, name, is_red, chessboard, fen)
        self._Chessman__top = 9
        self._Chessman__bottom = 0
        self._Chessman__left = 0
        self._Chessman__right = 8

    def calc_moving_list(self):
        current_v_c = super(Knight, self).position.x
        current_h_c = super(Knight, self).position.y
        probable_obstacle_points = []
        probable_moving_points = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c,)
        vs2 = (current_v_c,)
        hs2 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_obstacle_points, vs1, hs1)
        creat_points(probable_obstacle_points, vs2, hs2)
        current_color = super(Knight, self).is_red
        for point in probable_obstacle_points:
            if super(Knight, self).border_check(point.x, point.y):
                chessman = super(Knight, self).chessboard.get_chessman(
                    point.x, point.y)
                if chessman is None:
                    if point.x == current_v_c:
                        probable_moving_points.append(
                            Point(point.x + 1, 2 * point.y - current_h_c))
                        probable_moving_points.append(
                            Point(point.x - 1, 2 * point.y - current_h_c))
                    else:
                        probable_moving_points.append(
                            Point(2 * point.x - current_v_c, point.y + 1))
                        probable_moving_points.append(
                            Point(2 * point.x - current_v_c, point.y - 1))
        super(Knight, self).add_from_probable_points(
            probable_moving_points, current_color)


class Cannon(Chessman):
    '''炮'''

    def __init__(self, name_cn, name, is_red, chessboard, fen):
        super(Cannon, self).__init__(name_cn, name, is_red, chessboard, fen)
        self._Chessman__top = 9
        self._Chessman__bottom = 0
        self._Chessman__left = 0
        self._Chessman__right = 8

    def calc_moving_list(self):
        current_v_c = super(Cannon, self).position.x
        current_h_c = super(Cannon, self).position.y
        left = super(Cannon, self).chessboard.get_left_first_chessman(
            current_v_c, current_h_c)
        right = super(Cannon, self).chessboard.get_right_first_chessman(
            current_v_c, current_h_c)
        top = super(Cannon, self).chessboard.get_top_first_chessman(
            current_v_c, current_h_c)
        bottom = super(Cannon, self).chessboard.get_bottom_first_chessman(
            current_v_c, current_h_c)
        tar_left = super(Cannon, self).chessboard.get_left_second_chessman(
            current_v_c, current_h_c)
        tar_right = super(Cannon, self).chessboard.get_right_second_chessman(
            current_v_c, current_h_c)
        tar_top = super(Cannon, self).chessboard.get_top_second_chessman(
            current_v_c, current_h_c)
        tar_bottom = super(Cannon, self).chessboard.get_bottom_second_chessman(
            current_v_c, current_h_c)
        super(Cannon, self).calc_moving_path(left, (left.position.x if left != None else None),
                                             current_v_c, current_h_c, 1, 0, True, True)
        super(Cannon, self).calc_moving_path(right, (right.position.x if right != None else None),
                                             current_v_c, current_h_c, -1, 8, True, True)
        super(Cannon, self).calc_moving_path(top, (top.position.y if top != None else None),
                                             current_h_c, current_v_c, -1, 9, False, True)
        super(Cannon, self).calc_moving_path(bottom, (bottom.position.y if bottom != None else None),
                                             current_h_c, current_v_c, 1, 0, False, True)
        current_color = super(Cannon, self).is_red
        if tar_left != None and tar_left.is_red != current_color:
            super(Cannon, self).moving_list.append(
                Point(tar_left.position.x, tar_left.position.y))
        if tar_right != None and tar_right.is_red != current_color:
            super(Cannon, self).moving_list.append(
                Point(tar_right.position.x, tar_right.position.y))
        if tar_top != None and tar_top.is_red != current_color:
            super(Cannon, self).moving_list.append(
                Point(tar_top.position.x, tar_top.position.y))
        if tar_bottom != None and tar_bottom.is_red != current_color:
            super(Cannon, self).moving_list.append(
                Point(tar_bottom.position.x, tar_bottom.position.y))


class Mandarin(Chessman):
    '''仕/士'''

    def __init__(self, name_cn, name, is_red, chessboard, fen):
        super(Mandarin, self).__init__(name_cn, name, is_red, chessboard, fen)
        if self.is_red:
            self._Chessman__top = 2
            self._Chessman__bottom = 0
            self._Chessman__left = 3
            self._Chessman__right = 5
        else:
            self._Chessman__top = 9
            self._Chessman__bottom = 7
            self._Chessman__left = 3
            self._Chessman__right = 5

    def calc_moving_list(self):
        current_v_c = super(Mandarin, self).position.x
        current_h_c = super(Mandarin, self).position.y
        probable_moving_points = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_moving_points, vs1, hs1)
        current_color = super(Mandarin, self).is_red

        super(Mandarin, self).add_from_probable_points(
            probable_moving_points, current_color)


class Elephant(Chessman):
    '''象/相'''

    def __init__(self, name_cn, name, is_red, chessboard, fen):
        super(Elephant, self).__init__(name_cn, name, is_red, chessboard, fen)
        if self.is_red:
            self._Chessman__top = 4
            self._Chessman__bottom = 0
            self._Chessman__left = 0
            self._Chessman__right = 8
        else:
            self._Chessman__top = 9
            self._Chessman__bottom = 5
            self._Chessman__left = 0
            self._Chessman__right = 8

    def calc_moving_list(self):
        current_v_c = super(Elephant, self).position.x
        current_h_c = super(Elephant, self).position.y
        probable_obstacle_points = []
        probable_moving_points = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_obstacle_points, vs1, hs1)
        current_color = super(Elephant, self).is_red
        for point in probable_obstacle_points:
            if super(Elephant, self).border_check(point.x, point.y):
                chessman = super(Elephant, self).chessboard.get_chessman(
                    point.x, point.y)
                if chessman is None:
                    probable_moving_points.append(
                        Point(2 * point.x - current_v_c, 2 * point.y - current_h_c))
        super(Elephant, self).add_from_probable_points(
            probable_moving_points, current_color)


class Pawn(Chessman):
    '''卒/兵'''

    def __init__(self, name_cn, name, is_red, chessboard, fen):
        super(Pawn, self).__init__(name_cn, name, is_red, chessboard, fen)
        if self.is_red:
            self._Chessman__top = 9
            self._Chessman__bottom = 3
            self._Chessman__left = 0
            self._Chessman__right = 8
            self.__direction = 1
            self.__river = 5
        else:
            self._Chessman__top = 6
            self._Chessman__bottom = 0
            self._Chessman__left = 0
            self._Chessman__right = 8
            self.__direction = -1
            self.__river = 4

    def calc_moving_list(self):
        current_v_c = super(Pawn, self).position.x
        current_h_c = super(Pawn, self).position.y
        probable_moving_points = []
        current_color = super(Pawn, self).is_red
        probable_moving_points.append(
            Point(current_v_c, current_h_c + self.__direction))
        if current_h_c * self.__direction >= self.__river * self.__direction:
            probable_moving_points.append(
                Point(current_v_c + 1, current_h_c))
            probable_moving_points.append(
                Point(current_v_c - 1, current_h_c))
        super(Pawn, self).add_from_probable_points(
            probable_moving_points, current_color)


class King(Chessman):
    '''将/帅'''

    def __init__(self, name_cn, name, is_red, chessboard, fen):
        super(King, self).__init__(name_cn, name, is_red, chessboard, fen)
        if self.is_red:
            self._Chessman__top = 2
            self._Chessman__bottom = 0
            self._Chessman__left = 3
            self._Chessman__right = 5
        else:
            self._Chessman__top = 9
            self._Chessman__bottom = 7
            self._Chessman__left = 3
            self._Chessman__right = 5

    def calc_moving_list(self):
        current_v_c = super(King, self).position.x
        current_h_c = super(King, self).position.y
        probable_moving_points = []
        vs1 = (current_v_c + 1, current_v_c - 1)
        hs1 = (current_h_c,)
        vs2 = (current_v_c,)
        hs2 = (current_h_c + 1, current_h_c - 1)
        creat_points(probable_moving_points, vs1, hs1)
        creat_points(probable_moving_points, vs2, hs2)
        current_color = super(King, self).is_red
        super(King, self).add_from_probable_points(
            probable_moving_points, current_color)
