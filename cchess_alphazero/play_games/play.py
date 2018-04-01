import sys
import pygame
import random
import os.path
import time
import numpy as np

from pygame.locals import *
from logging import getLogger
from collections import defaultdict
from threading import Thread

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.environment.chessboard import Chessboard
from cchess_alphazero.environment.chessman import *
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_move
from cchess_alphazero.lib.model_helper import load_best_model_weight

logger = getLogger(__name__)
main_dir = os.path.split(os.path.abspath(__file__))[0]
PIECE_STYLE = 'WOOD'

def start(config: Config, human_move_first=True):
    global PIECE_STYLE
    PIECE_STYLE = config.opts.piece_style
    play = PlayWithHuman(config)
    play.start(human_move_first)

class PlayWithHuman:
    def __init__(self, config: Config):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.winstyle = pygame.RESIZABLE
        self.chessmans = None
        self.human_move_first = True
        self.height = 577
        self.width = 521
        self.chessman_w = 57
        self.chessman_h = 57
        if self.config.opts.bg_style == 'WOOD':
            self.chessman_w += 1
            self.chessman_h += 1

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def init_screen(self):
        bestdepth = pygame.display.mode_ok([self.width, self.height], self.winstyle, 32)
        screen = pygame.display.set_mode([self.width, self.height], self.winstyle, bestdepth)
        pygame.display.set_caption("中国象棋-AlphaHe")
        # create the background, tile the bgd image
        bgdtile = load_image(f'{self.config.opts.bg_style}.gif')
        bgdtile = pygame.transform.scale(bgdtile, (self.width, self.height))
        background = pygame.Surface([self.width, self.height])
        for x in range(0, self.width, bgdtile.get_width()):
            background.blit(bgdtile, (x, 0))
        screen.blit(background, (0, 0))
        pygame.display.flip()
        self.chessmans = pygame.sprite.Group()
        creat_sprite_group(self.chessmans, self.env.board.chessmans_hash, self.chessman_w, self.chessman_h)
        return screen, background

    def start(self, human_first=True):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=True)
        self.human_move_first = human_first

        pygame.init()
        screen, background = self.init_screen()
        framerate = pygame.time.Clock()

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        current_chessman = None
        if human_first:
            self.env.board.calc_chessmans_moving_list()

        ai_worker = Thread(target=self.ai_move, name="ai_worker")
        ai_worker.daemon = True
        ai_worker.start()

        while not self.env.board.is_end():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.board.print_record()
                    self.ai.close()
                    sys.exit()
                elif event.type == VIDEORESIZE:
                    self.chessman_w = int(self.chessman_w * event.w / self.width)
                    self.chessman_h = int(self.chessman_h * event.h / self.height)
                    self.width = event.w
                    self.height = event.h
                    screen, background = self.init_screen()
                elif event.type == MOUSEBUTTONDOWN:
                    if human_first == self.env.red_to_move:
                        pressed_array = pygame.mouse.get_pressed()
                        for index in range(len(pressed_array)):
                            if index == 0 and pressed_array[index]:
                                mouse_x, mouse_y = pygame.mouse.get_pos()
                                col_num, row_num = translate_hit_area(mouse_x, mouse_y, self.chessman_w, self.chessman_h)
                                chessman_sprite = select_sprite_from_group(
                                    self.chessmans, col_num, row_num)
                                if current_chessman is None and chessman_sprite != None:
                                    if chessman_sprite.chessman.is_red == self.env.red_to_move:
                                        current_chessman = chessman_sprite
                                        chessman_sprite.is_selected = True
                                elif current_chessman != None and chessman_sprite != None:
                                    if chessman_sprite.chessman.is_red == self.env.red_to_move:
                                        current_chessman.is_selected = False
                                        current_chessman = chessman_sprite
                                        chessman_sprite.is_selected = True
                                    else:
                                        success = current_chessman.move(col_num, row_num, self.chessman_w, self.chessman_h)
                                        if success:
                                            self.chessmans.remove(chessman_sprite)
                                            chessman_sprite.kill()
                                            current_chessman.is_selected = False
                                            current_chessman = None
                                elif current_chessman != None and chessman_sprite is None:
                                    success = current_chessman.move(col_num, row_num, self.chessman_w, self.chessman_h)
                                    if success:
                                        current_chessman.is_selected = False
                                        current_chessman = None
                          
            framerate.tick(20)
            # clear/erase the last drawn sprites
            self.chessmans.clear(screen, background)

            # update all the sprites
            self.chessmans.update()
            self.chessmans.draw(screen)
            pygame.display.update()

        self.ai.close()
        logger.info(f"Winner is {self.env.board.winner} !!!")
        self.env.board.print_record()

    def ai_move(self):
        ai_move_first = not self.human_move_first
        while not self.env.done:
            if ai_move_first == self.env.red_to_move:
                labels = ActionLabelsRed
                labels_n = len(ActionLabelsRed)
                self.ai.search_results = {}
                action, policy = self.ai.action(self.env.get_state(), self.env.num_halfmoves)
                if not self.env.red_to_move:
                    action = flip_move(action)
                if action is None:
                    logger.info("AI has resigned!")
                    return
                key = self.env.get_state()
                p, v = self.ai.debug[key]
                logger.info(f"NN value = {v:.3f}")
                logger.info("MCTS results:")
                for move, action_state in self.ai.search_results.items():
                    if action_state[0] > 20:
                        move_cn = self.env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
                        logger.info(f"move: {move_cn}-{move}, visit count: {action_state[0]}, Q_value: {action_state[1]:.3f}, Prior: {action_state[2]:.3f}")
                x0, y0, x1, y1 = int(action[0]), int(action[1]), int(action[2]), int(action[3])
                chessman_sprite = select_sprite_from_group(self.chessmans, x0, y0)
                sprite_dest = select_sprite_from_group(self.chessmans, x1, y1)
                if sprite_dest:
                    self.chessmans.remove(sprite_dest)
                    sprite_dest.kill()
                chessman_sprite.move(x1, y1, self.chessman_w, self.chessman_h)
        

class Chessman_Sprite(pygame.sprite.Sprite):
    is_selected = False
    images = []
    is_transparent = False

    def __init__(self, images, chessman, w=80, h=80):
        pygame.sprite.Sprite.__init__(self)
        self.chessman = chessman
        self.images = [pygame.transform.scale(image, (w, h)) for image in images]
        self.image = self.images[0]
        self.rect = Rect(chessman.col_num * w, (9 - chessman.row_num) * h, w, h)

    def move(self, col_num, row_num, w=80, h=80):
        # print self.chessman.name, col_num, row_num
        old_col_num = self.chessman.col_num
        old_row_num = self.chessman.row_num
        is_correct_position = self.chessman.move(col_num, row_num)
        if is_correct_position:
            self.rect = Rect(old_col_num * w, (9 - old_row_num) * h, w, h)
            self.rect.move_ip((col_num - old_col_num)
                              * w, (old_row_num - row_num) * h)
            # self.rect = self.rect.clamp(SCREENRECT)
            self.chessman.chessboard.clear_chessmans_moving_list()
            self.chessman.chessboard.calc_chessmans_moving_list()
            return True
        return False

    def update(self):
        if self.is_selected:
            self.image = self.images[1]
        else:
            self.image = self.images[0]


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
    global PIECE_STYLE
    imgs = []
    for file in files:
        imgs.append(load_image(file, PIECE_STYLE))
    return imgs

def creat_sprite_group(sprite_group, chessmans_hash, w, h):
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
        chessman_sprite = Chessman_Sprite(images, chess, w, h)
        sprite_group.add(chessman_sprite)

def select_sprite_from_group(sprite_group, col_num, row_num):
    for sprite in sprite_group:
        if sprite.chessman.col_num == col_num and sprite.chessman.row_num == row_num:
            return sprite
    return None

def translate_hit_area(screen_x, screen_y, w=80, h = 80):
    return screen_x // w, 9 - screen_y // h

