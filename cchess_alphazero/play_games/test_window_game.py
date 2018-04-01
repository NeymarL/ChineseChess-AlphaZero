#-*- coding:utf-8 -*-

import sys
import pygame
import random
import os.path

from cchess_alphazero.environment.chessboard import Chessboard
from cchess_alphazero.environment.chessman import *
from pygame.locals import *

main_dir = os.path.split(os.path.abspath(__file__))[0]
SCREENRECT = Rect(0, 0, 700, 577)
PIECE_STYLE = 'POLISH'
BOARD_STYLE = 'QIANHONG'


def load_image(file, sub_dir=None):
    '''loads an image, prepares it for play'''
    if sub_dir:
        file = os.path.join(main_dir, 'images', sub_dir, file)
    else:
        file = os.path.join(main_dir, 'images', file)
    try:
        surface = pygame.image.load(file)
    except pygame.error:
        raise SystemExit('Could not load image "%s" %s' %
                         (file, pygame.get_error()))
    return surface.convert()


def load_images(*files):
    imgs = []
    style = PIECE_STYLE
    for file in files:
        imgs.append(load_image(file, style))
    return imgs


class Chessman_Sprite(pygame.sprite.Sprite):
    is_selected = False
    images = []
    is_transparent = False

    def __init__(self, images, chessman):
        pygame.sprite.Sprite.__init__(self)
        self.chessman = chessman
        self.images = images
        self.image = self.images[0]
        self.rect = Rect(chessman.col_num * 57, (9 - chessman.row_num) * 57, 57, 57)

    def move(self, col_num, row_num):
        # print self.chessman.name, col_num, row_num
        old_col_num = self.chessman.col_num
        old_row_num = self.chessman.row_num
        is_correct_position = self.chessman.move(col_num, row_num)
        if is_correct_position:
            self.rect.move_ip((col_num - old_col_num)
                              * 57, (old_row_num - row_num) * 57)
            self.rect = self.rect.clamp(SCREENRECT)
            self.chessman.chessboard.clear_chessmans_moving_list()
            self.chessman.chessboard.calc_chessmans_moving_list()
            return True
        return False

    def update(self):
        if self.is_selected:
            self.image = self.images[1]
        else:
            self.image = self.images[0]


def creat_sprite_group(sprite_group, chessmans_hash):
    for chess in chessmans_hash.values():
        if chess.is_red:
            if isinstance(chess, Rook):
                images = load_images("RR.gif", "RRS.gif")
            elif isinstance(chess, Cannon):
                images = load_images("RC.gif", "RCS.gif")
            elif isinstance(chess, Knight):
                images = load_images("RN.gif", "RNS.gif")
            elif isinstance(chess, King):
                images = load_images("RK.gif", "RKS.gif")
            elif isinstance(chess, Elephant):
                images = load_images("RB.gif", "RBS.gif")
            elif isinstance(chess, Mandarin):
                images = load_images("RA.gif", "RAS.gif")
            else:
                images = load_images("RP.gif", "RPS.gif")
        else:
            if isinstance(chess, Rook):
                images = load_images("BR.gif", "BRS.gif")
            elif isinstance(chess, Cannon):
                images = load_images("BC.gif", "BCS.gif")
            elif isinstance(chess, Knight):
                images = load_images("BN.gif", "BNS.gif")
            elif isinstance(chess, King):
                images = load_images("BK.gif", "BKS.gif")
            elif isinstance(chess, Elephant):
                images = load_images("BB.gif", "BBS.gif")
            elif isinstance(chess, Mandarin):
                images = load_images("BA.gif", "BAS.gif")
            else:
                images = load_images("BP.gif", "BPS.gif")
        chessman_sprite = Chessman_Sprite(images, chess)
        sprite_group.add(chessman_sprite)


def select_sprite_from_group(sprite_group, col_num, row_num):
    for sprite in sprite_group:
        if sprite.chessman.col_num == col_num and sprite.chessman.row_num == row_num:
            return sprite


def translate_hit_area(screen_x, screen_y):
    return screen_x // 57, 9 - screen_y // 57


def main(winstyle=0):

    pygame.init()
    bestdepth = pygame.display.mode_ok(SCREENRECT.size, winstyle, 32)
    screen = pygame.display.set_mode(SCREENRECT.size, winstyle, bestdepth)
    pygame.display.set_caption("中国象棋-AlphaZero")

    # create the background, tile the bgd image
    bgdtile = load_image(f'{BOARD_STYLE}.gif')
    board_background = pygame.Surface([521, 577])
    board_background.blit(bgdtile, (0, 0))
    widget_background = pygame.Surface([700 - 521, 577])
    white_rect = Rect(0, 0, 700 - 521, 577)
    widget_background.fill((255, 255, 255), white_rect)

    #create text label
    font_file = os.path.join(main_dir, 'PingFang.ttc')
    font = pygame.font.Font(font_file, 16)
    font_color = (0, 0, 0)
    font_background = (255, 255, 255)
    t = font.render("着法记录", True, font_color, font_background)
    t_rect = t.get_rect()
    t_rect.centerx = (700 - 521) / 2
    t_rect.y = 10
    widget_background.blit(t, t_rect)

    # background = pygame.Surface(SCREENRECT.size)
    # for x in range(0, SCREENRECT.width, bgdtile.get_width()):
    #     background.blit(bgdtile, (x, 0))
    screen.blit(board_background, (0, 0))
    screen.blit(widget_background, (521, 0))
    pygame.display.flip()

    cbd = Chessboard('000')
    cbd.init_board()

    chessmans = pygame.sprite.Group()
    framerate = pygame.time.Clock()

    creat_sprite_group(chessmans, cbd.chessmans_hash)
    current_chessman = None
    cbd.calc_chessmans_moving_list()
    print(cbd.legal_moves())
    while not cbd.is_end():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cbd.print_record()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                pressed_array = pygame.mouse.get_pressed()
                for index in range(len(pressed_array)):
                    if index == 0 and pressed_array[index]:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        col_num, row_num = translate_hit_area(mouse_x, mouse_y)
                        chessman_sprite = select_sprite_from_group(
                            chessmans, col_num, row_num)
                        if current_chessman is None and chessman_sprite != None:
                            if chessman_sprite.chessman.is_red == cbd.is_red_turn:
                                current_chessman = chessman_sprite
                                chessman_sprite.is_selected = True
                        elif current_chessman != None and chessman_sprite != None:
                            if chessman_sprite.chessman.is_red == cbd.is_red_turn:
                                current_chessman.is_selected = False
                                current_chessman = chessman_sprite
                                chessman_sprite.is_selected = True
                            else:
                                success = current_chessman.move(col_num, row_num)
                                if success:
                                    chessmans.remove(chessman_sprite)
                                    chessman_sprite.kill()
                                    current_chessman.is_selected = False
                                    current_chessman = None
                                    print(cbd.legal_moves())
                        elif current_chessman != None and chessman_sprite is None:
                            success = current_chessman.move(col_num, row_num)
                            if success:
                                current_chessman.is_selected = False
                                current_chessman = None
                                print(cbd.legal_moves())
        records = cbd.record.split('\n')
        font = pygame.font.Font(font_file, 12)
        i = 0
        for record in records[-20:]:
            rec_label = font.render(record, True, font_color, font_background)
            t_rect = rec_label.get_rect()
            t_rect.centerx = (700 - 521) / 2
            t_rect.y = 35 + i * 15
            widget_background.blit(rec_label, t_rect)
            i += 1
        screen.blit(widget_background, (521, 0))
        framerate.tick(20)
        # clear/erase the last drawn sprites
        chessmans.clear(screen, board_background)

        # update all the sprites
        chessmans.update()
        chessmans.draw(screen)
        pygame.display.update()

    cbd.print_record()

if __name__ == '__main__':
    main()
