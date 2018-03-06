#-*- coding:utf-8 -*-

import chessboard
import chessman

def print_chessman_name(chessman):
    if chessman:
        print(chessman.name)
    else:
        print("None")


def main():
    cbd = chessboard.Chessboard('000')
    cbd.init_board()
    cbd.print_to_cl()
    # cbd.remove_chessman_source(0,0)
    # cbd.print_to_cl()
    while not cbd.is_end():
        cbd.calc_chessmans_moving_list()
        if cbd.is_red_turn:
            print("is_red_turn")
        else:
            print("is_black_turn")
        is_correct_chessman = False
        is_correct_position = False
        chessman = None
        while not is_correct_chessman:
            title = "请输入棋子名字: "
            input_chessman_name = input(title)
            chessman = cbd.get_chessman_by_name(input_chessman_name)
            if chessman != None and chessman.is_red == cbd.is_red_turn:
                is_correct_chessman = True
                print("当前可以落子的位置有：")
                for point in chessman.moving_list:
                    print(point.x, point.y)
            else:
                print("没有找到此名字的棋子或未轮到此方走子")
        while not is_correct_position:
            title = "请输入落子的位置: "
            input_chessman_position0 = int(input(title))
            input_chessman_position1 = int(input(title))
            print("Inputed Position:", input_chessman_position0, input_chessman_position1)
            is_correct_position = chessman.move(
                input_chessman_position0, input_chessman_position1)
            if is_correct_position:
                cbd.print_to_cl()
                cbd.clear_chessmans_moving_list()


if __name__ == '__main__':
    main()

