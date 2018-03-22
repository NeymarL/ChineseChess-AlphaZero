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

from cchess_alphazero.environment.chessboard import Chessboard
from cchess_alphazero.environment.chessman import *
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed
from cchess_alphazero.lib.model_helper import load_best_model_weight

logger = getLogger(__name__)
main_dir = os.path.split(os.path.abspath(__file__))[0]
SCREENRECT = Rect(0, 0, 720, 800)

def start(config: Config, human_move_first=True):
    play = PlayWithHuman(config)
    play.start(human_move_first)

class PlayWithHuman:
    def __init__(self, config: Config):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.winstyle = 0
        self.chessmans = None
        self.human_move_first = True

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def start(self, human_first=True):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes(self.config.play.search_threads)
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=True)
        self.human_move_first = human_first

        pygame.init()
        bestdepth = pygame.display.mode_ok(SCREENRECT.size, self.winstyle, 32)
        screen = pygame.display.set_mode(SCREENRECT.size, self.winstyle, bestdepth)
        pygame.display.set_caption("中国象棋-AlphaZero")
        # create the background, tile the bgd image
        bgdtile = load_image('boardchess.gif')
        background = pygame.Surface(SCREENRECT.size)
        for x in range(0, SCREENRECT.width, bgdtile.get_width()):
            background.blit(bgdtile, (x, 0))
        screen.blit(background, (0, 0))
        pygame.display.flip()

        self.chessmans = pygame.sprite.Group()
        framerate = pygame.time.Clock()

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        creat_sprite_group(self.chessmans, self.env.board.chessmans_hash)
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
                    sys.exit()
                elif event.type == MOUSEBUTTONDOWN:
                    if human_first == self.env.red_to_move:
                        pressed_array = pygame.mouse.get_pressed()
                        for index in range(len(pressed_array)):
                            if index == 0 and pressed_array[index]:
                                mouse_x, mouse_y = pygame.mouse.get_pos()
                                col_num, row_num = translate_hit_area(mouse_x, mouse_y)
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
                                        success = current_chessman.move(col_num, row_num)
                                        if success:
                                            self.chessmans.remove(chessman_sprite)
                                            chessman_sprite.kill()
                                            current_chessman.is_selected = False
                                            current_chessman = None
                                elif current_chessman != None and chessman_sprite is None:
                                    success = current_chessman.move(col_num, row_num)
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

        logger.info(f"Winner is {self.env.board.winner} !!!")
        self.env.board.print_record()

    def ai_move(self):
        ai_move_first = not self.human_move_first
        while not self.env.done:
            if ai_move_first == self.env.red_to_move:
                labels = ActionLabelsRed
                labels_n = len(ActionLabelsRed)
                self.ai.search_results = {}
                action = self.ai.action(self.env)
                if action is None:
                    logger.info("AI has resigned!")
                    return
                key = self.ai.get_state_key(self.env)
                p, v = self.ai.debug[key]
                mov_idx = np.argmax(p)
                mov = labels[mov_idx]
                mov = self.env.board.make_single_record(int(mov[0]), int(mov[1]), int(mov[2]), int(mov[3]))
                logger.info(f"NN value = {v:.2f}")
                logger.info("MCTS results:")
                for move, action_state in self.ai.search_results.items():
                    if action_state[0] >= 15:
                        move = self.env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
                        logger.info(f"move: {move}, prob: {action_state[0]}, Q_value: {action_state[1]:.2f}, Prior: {action_state[2]:.3f}")
                
                x0, y0, x1, y1 = int(action[0]), int(action[1]), int(action[2]), int(action[3])
                chessman_sprite = select_sprite_from_group(self.chessmans, x0, y0)
                sprite_dest = select_sprite_from_group(self.chessmans, x1, y1)
                if sprite_dest:
                    self.chessmans.remove(sprite_dest)
                    sprite_dest.kill()
                chessman_sprite.move(x1, y1)
        

class Chessman_Sprite(pygame.sprite.Sprite):
    is_selected = False
    images = []
    is_transparent = False

    def __init__(self, images, chessman):
        pygame.sprite.Sprite.__init__(self)
        self.chessman = chessman
        self.images = images
        self.image = self.images[0]
        self.rect = Rect(chessman.col_num * 80, (9 - chessman.row_num) * 80, 80, 80)

    def move(self, col_num, row_num):
        # print self.chessman.name, col_num, row_num
        old_col_num = self.chessman.col_num
        old_row_num = self.chessman.row_num
        is_correct_position = self.chessman.move(col_num, row_num)
        if is_correct_position:
            self.rect.move_ip((col_num - old_col_num)
                              * 80, (old_row_num - row_num) * 80)
            self.rect = self.rect.clamp(SCREENRECT)
            self.chessman.chessboard.clear_chessmans_moving_list()
            self.chessman.chessboard.calc_chessmans_moving_list()
            return True
        return False

    def update(self):
        if self.is_selected:
            if self.is_transparent:
                self.image = self.images[1]
            else:
                self.image = self.images[0]
            self.is_transparent = not self.is_transparent
        else:
            self.image = self.images[0]


def load_image(file):
    '''loads an image, prepares it for play'''
    file = os.path.join(main_dir, 'images', file)
    try:
        surface = pygame.image.load(file)
    except pygame.error:
        raise SystemExit('Could not load image "%s" %s' %
                         (file, pygame.get_error()))
    return surface.convert()

def load_images(*files):
    imgs = []
    for file in files:
        imgs.append(load_image(file))
    return imgs

def creat_sprite_group(sprite_group, chessmans_hash):
    for chess in chessmans_hash.values():
        if chess.is_red:
            if isinstance(chess, Rook):
                images = load_images("red_rook.gif", "transparent.gif")
            elif isinstance(chess, Cannon):
                images = load_images("red_cannon.gif", "transparent.gif")
            elif isinstance(chess, Knight):
                images = load_images("red_knight.gif", "transparent.gif")
            elif isinstance(chess, King):
                images = load_images("red_king.gif", "transparent.gif")
            elif isinstance(chess, Elephant):
                images = load_images("red_elephant.gif", "transparent.gif")
            elif isinstance(chess, Mandarin):
                images = load_images("red_mandarin.gif", "transparent.gif")
            else:
                images = load_images("red_pawn.gif", "transparent.gif")
        else:
            if isinstance(chess, Rook):
                images = load_images("black_rook.gif", "transparent.gif")
            elif isinstance(chess, Cannon):
                images = load_images("black_cannon.gif", "transparent.gif")
            elif isinstance(chess, Knight):
                images = load_images("black_knight.gif", "transparent.gif")
            elif isinstance(chess, King):
                images = load_images("black_king.gif", "transparent.gif")
            elif isinstance(chess, Elephant):
                images = load_images("black_elephant.gif", "transparent.gif")
            elif isinstance(chess, Mandarin):
                images = load_images("black_mandarin.gif", "transparent.gif")
            else:
                images = load_images("black_pawn.gif", "transparent.gif")
        chessman_sprite = Chessman_Sprite(images, chess)
        sprite_group.add(chessman_sprite)

def select_sprite_from_group(sprite_group, col_num, row_num):
    for sprite in sprite_group:
        if sprite.chessman.col_num == col_num and sprite.chessman.row_num == row_num:
            return sprite
    return None

def translate_hit_area(screen_x, screen_y):
    return screen_x // 80, 9 - screen_y // 80

