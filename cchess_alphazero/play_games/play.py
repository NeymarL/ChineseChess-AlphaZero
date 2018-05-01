import sys
import pygame
import random
import os.path
import time
import copy
import numpy as np

from pygame.locals import *
from logging import getLogger
from collections import defaultdict
from threading import Thread
from time import sleep
from datetime import datetime

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
from cchess_alphazero.lib.tf_util import set_session_config

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
        self.winstyle = 0
        self.chessmans = None
        self.human_move_first = True
        self.screen_width = 720
        self.height = 577
        self.width = 521
        self.chessman_w = 57
        self.chessman_h = 57
        self.disp_record_num = 15
        self.rec_labels = [None] * self.disp_record_num
        self.nn_value = 0
        self.mcts_moves = {}
        self.history = []
        if self.config.opts.bg_style == 'WOOD':
            self.chessman_w += 1
            self.chessman_h += 1

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def init_screen(self):
        bestdepth = pygame.display.mode_ok([self.screen_width, self.height], self.winstyle, 32)
        screen = pygame.display.set_mode([self.screen_width, self.height], self.winstyle, bestdepth)
        pygame.display.set_caption("中国象棋-AlphaZero")
        # create the background, tile the bgd image
        bgdtile = load_image(f'{self.config.opts.bg_style}.GIF')
        bgdtile = pygame.transform.scale(bgdtile, (self.width, self.height))
        board_background = pygame.Surface([self.width, self.height])
        board_background.blit(bgdtile, (0, 0))
        widget_background = pygame.Surface([self.screen_width - self.width, self.height])
        white_rect = Rect(0, 0, self.screen_width - self.width, self.height)
        widget_background.fill((255, 255, 255), white_rect)

        #create text label
        font_file = self.config.resource.font_path
        font = pygame.font.Font(font_file, 16)
        font_color = (0, 0, 0)
        font_background = (255, 255, 255)
        t = font.render("着法记录", True, font_color, font_background)
        t_rect = t.get_rect()
        t_rect.x = 10
        t_rect.y = 10
        widget_background.blit(t, t_rect)

        screen.blit(board_background, (0, 0))
        screen.blit(widget_background, (self.width, 0))
        pygame.display.flip()
        self.chessmans = pygame.sprite.Group()
        creat_sprite_group(self.chessmans, self.env.board.chessmans_hash, self.chessman_w, self.chessman_h)
        return screen, board_background, widget_background

    def start(self, human_first=True):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=True)
        self.human_move_first = human_first

        pygame.init()
        screen, board_background, widget_background = self.init_screen()
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
                    self.ai.close(wait=False)
                    game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                    path = os.path.join(self.config.resource.play_record_dir, self.config.resource.play_record_filename_tmpl % game_id)
                    self.env.board.save_record(path)
                    sys.exit()
                elif event.type == VIDEORESIZE:
                    pass
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
                                        move = str(current_chessman.chessman.col_num) + str(current_chessman.chessman.row_num) +\
                                               str(col_num) + str(row_num)
                                        success = current_chessman.move(col_num, row_num, self.chessman_w, self.chessman_h)
                                        self.history.append(move)
                                        if success:
                                            self.chessmans.remove(chessman_sprite)
                                            chessman_sprite.kill()
                                            current_chessman.is_selected = False
                                            current_chessman = None
                                            self.history.append(self.env.get_state())
                                elif current_chessman != None and chessman_sprite is None:
                                    move = str(current_chessman.chessman.col_num) + str(current_chessman.chessman.row_num) +\
                                           str(col_num) + str(row_num)
                                    success = current_chessman.move(col_num, row_num, self.chessman_w, self.chessman_h)
                                    self.history.append(move)
                                    if success:
                                        current_chessman.is_selected = False
                                        current_chessman = None
                                        self.history.append(self.env.get_state())

            self.draw_widget(screen, widget_background)
            framerate.tick(20)
            # clear/erase the last drawn sprites
            self.chessmans.clear(screen, board_background)

            # update all the sprites
            self.chessmans.update()
            self.chessmans.draw(screen)
            pygame.display.update()

        self.ai.close(wait=False)
        logger.info(f"Winner is {self.env.board.winner} !!!")
        self.env.board.print_record()
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.config.resource.play_record_dir, self.config.resource.play_record_filename_tmpl % game_id)
        self.env.board.save_record(path)
        sleep(3)

    def ai_move(self):
        ai_move_first = not self.human_move_first
        self.history = [self.env.get_state()]
        no_act = None
        while not self.env.done:
            if ai_move_first == self.env.red_to_move:
                labels = ActionLabelsRed
                labels_n = len(ActionLabelsRed)
                self.ai.search_results = {}
                state = self.env.get_state()
                logger.info(f"state = {state}")
                _, _, _, check = senv.done(state, need_check=True)
                if not check and state in self.history[:-1]:
                    no_act = []
                    free_move = defaultdict(int)
                    for i in range(len(self.history) - 1):
                        if self.history[i] == state:
                            # 如果走了下一步是将军或捉：禁止走那步
                            if senv.will_check_or_catch(state, self.history[i+1]):
                                no_act.append(self.history[i + 1])
                            # 否则当作闲着处理
                            else:
                                free_move[state] += 1
                                if free_move[state] >= 2:
                                    # 作和棋处理
                                    self.env.winner = Winner.draw
                                    self.env.board.winner = Winner.draw
                                    break
                    if no_act:
                        logger.debug(f"no_act = {no_act}")
                action, policy = self.ai.action(state, self.env.num_halfmoves, no_act)
                if action is None:
                    logger.info("AI has resigned!")
                    return
                self.history.append(action)
                if not self.env.red_to_move:
                    action = flip_move(action)
                key = self.env.get_state()
                p, v = self.ai.debug[key]
                logger.info(f"check = {check}, NN value = {v:.3f}")
                self.nn_value = v
                logger.info("MCTS results:")
                self.mcts_moves = {}
                for move, action_state in self.ai.search_results.items():
                    move_cn = self.env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
                    logger.info(f"move: {move_cn}-{move}, visit count: {action_state[0]}, Q_value: {action_state[1]:.3f}, Prior: {action_state[2]:.3f}")
                    self.mcts_moves[move_cn] = action_state
                x0, y0, x1, y1 = int(action[0]), int(action[1]), int(action[2]), int(action[3])
                chessman_sprite = select_sprite_from_group(self.chessmans, x0, y0)
                sprite_dest = select_sprite_from_group(self.chessmans, x1, y1)
                if sprite_dest:
                    self.chessmans.remove(sprite_dest)
                    sprite_dest.kill()
                chessman_sprite.move(x1, y1, self.chessman_w, self.chessman_h)
                self.history.append(self.env.get_state())

    def draw_widget(self, screen, widget_background):
        white_rect = Rect(0, 0, self.screen_width - self.width, self.height)
        widget_background.fill((255, 255, 255), white_rect)
        pygame.draw.line(widget_background, (255, 0, 0), (10, 285), (self.screen_width - self.width - 10, 285))
        screen.blit(widget_background, (self.width, 0))
        self.draw_records(screen, widget_background)
        self.draw_evaluation(screen, widget_background) 

    def draw_records(self, screen, widget_background):
        text = '着法记录'
        self.draw_label(screen, widget_background, text, 10, 16, 10)
        records = self.env.board.record.split('\n')
        font_file = self.config.resource.font_path
        font = pygame.font.Font(font_file, 12)
        i = 0
        for record in records[-self.disp_record_num:]:
            self.rec_labels[i] = font.render(record, True, (0, 0, 0), (255, 255, 255))
            t_rect = self.rec_labels[i].get_rect()
            # t_rect.centerx = (self.screen_width - self.width) / 2
            t_rect.y = 35 + i * 15
            t_rect.x = 10
            t_rect.width = self.screen_width - self.width
            widget_background.blit(self.rec_labels[i], t_rect)
            i += 1
        screen.blit(widget_background, (self.width, 0))

    def draw_evaluation(self, screen, widget_background):
        title_label = 'AlphaZero信息'
        self.draw_label(screen, widget_background, title_label, 300, 16, 10)
        info_label = f'MCTS搜索次数：{self.config.play.simulation_num_per_move}'
        self.draw_label(screen, widget_background, info_label, 335, 14, 10)
        eval_label = f"当前局势评估: {self.nn_value:.3f}"
        self.draw_label(screen, widget_background, eval_label, 360, 14, 10)
        label = f"MCTS搜索结果:"
        self.draw_label(screen, widget_background, label, 395, 14, 10)
        label = f"着法 访问计数 动作价值 先验概率"
        self.draw_label(screen, widget_background, label, 415, 12, 10)
        i = 0
        tmp = copy.deepcopy(self.mcts_moves)
        for mov, action_state in tmp.items():
            label = f"{mov}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 10)
            label = f"{action_state[0]}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 70)
            label = f"{action_state[1]:.2f}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 100)
            label = f"{action_state[2]:.3f}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 150)
            i += 1

    def draw_label(self, screen, widget_background, text, y, font_size, x=None):
        font_file = self.config.resource.font_path
        font = pygame.font.Font(font_file, font_size)
        label = font.render(text, True, (0, 0, 0), (255, 255, 255))
        t_rect = label.get_rect()
        t_rect.y = y
        if x != None:
            t_rect.x = x
        else:
            t_rect.centerx = (self.screen_width - self.width) / 2
        widget_background.blit(label, t_rect)
        screen.blit(widget_background, (self.width, 0))
        

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
                images = load_images("RR.GIF", "RRS.GIF")
            elif isinstance(chess, Cannon):
                images = load_images("RC.GIF", "RCS.GIF")
            elif isinstance(chess, Knight):
                images = load_images("RN.GIF", "RNS.GIF")
            elif isinstance(chess, King):
                images = load_images("RK.GIF", "RKS.GIF")
            elif isinstance(chess, Elephant):
                images = load_images("RB.GIF", "RBS.GIF")
            elif isinstance(chess, Mandarin):
                images = load_images("RA.GIF", "RAS.GIF")
            else:
                images = load_images("RP.GIF", "RPS.GIF")
        else:
            if isinstance(chess, Rook):
                images = load_images("BR.GIF", "BRS.GIF")
            elif isinstance(chess, Cannon):
                images = load_images("BC.GIF", "BCS.GIF")
            elif isinstance(chess, Knight):
                images = load_images("BN.GIF", "BNS.GIF")
            elif isinstance(chess, King):
                images = load_images("BK.GIF", "BKS.GIF")
            elif isinstance(chess, Elephant):
                images = load_images("BB.GIF", "BBS.GIF")
            elif isinstance(chess, Mandarin):
                images = load_images("BA.GIF", "BAS.GIF")
            else:
                images = load_images("BP.GIF", "BPS.GIF")
        chessman_sprite = Chessman_Sprite(images, chess, w, h)
        sprite_group.add(chessman_sprite)

def select_sprite_from_group(sprite_group, col_num, row_num):
    for sprite in sprite_group:
        if sprite.chessman.col_num == col_num and sprite.chessman.row_num == row_num:
            return sprite
    return None

def translate_hit_area(screen_x, screen_y, w=80, h = 80):
    return screen_x // w, 9 - screen_y // h

