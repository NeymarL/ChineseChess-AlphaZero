import os
import numpy as np
import pandas as pd

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from time import sleep
from random import shuffle
from time import time

from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.config import Config
from cchess_alphazero.lib.data_helper import get_game_data_filenames, read_game_data_from_file
from cchess_alphazero.lib.model_helper import load_sl_best_model_weight, save_as_sl_best_model
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, flip_policy, flip_move
from cchess_alphazero.lib.tf_util import set_session_config

from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import keras.backend as K

logger = getLogger(__name__)

def start(config: Config):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list='0,1')
    return SupervisedWorker(config).start()

class SupervisedWorker:
    def __init__(self, config:Config):
        self.config = config
        self.model = None
        self.loaded_data = deque(maxlen=self.config.trainer.dataset_size)
        self.dataset = deque(), deque(), deque()
        self.filenames = []
        self.opt = None
        self.buffer = []
        self.gameinfo = None
        self.moves = None
        self.config.opts.light = True

    def start(self):
        self.model = self.load_model()
        self.gameinfo = pd.read_csv(self.config.resource.sl_data_gameinfo)
        self.moves = pd.read_csv(self.config.resource.sl_data_move)
        self.training()

    def training(self):
        self.compile_model()
        total_steps = self.config.trainer.start_total_steps
        logger.info(f"Start training, game count = {len(self.gameinfo)}, step = {self.config.trainer.sl_game_step} games")

        for i in range(0, len(self.gameinfo), self.config.trainer.sl_game_step):
            games = self.gameinfo[i:i+self.config.trainer.sl_game_step]
            self.fill_queue(games)
            if len(self.dataset[0]) > self.config.trainer.batch_size:
                steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
                total_steps += steps
                self.save_current_model()
                a, b, c = self.dataset
                a.clear()
                b.clear()
                c.clear()
                self.update_learning_rate(total_steps)

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, value_ary = self.collect_all_loaded_data()
        tensorboard_cb = TensorBoard(log_dir="./logs/tensorboard_sl/", batch_size=tc.batch_size, histogram_freq=1)
        self.model.model.fit(state_ary, [policy_ary, value_ary],
                             batch_size=tc.batch_size,
                             epochs=epochs,
                             shuffle=True,
                             validation_split=0.02,
                             callbacks=[tensorboard_cb])
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        self.opt = Adam(lr=1e-2)
        losses = ['categorical_crossentropy', 'mean_squared_error'] # avoid overfit for supervised 
        self.model.model.compile(optimizer=self.opt, loss=losses, loss_weights=self.config.trainer.loss_weights)

    def fill_queue(self, games):
        _tuple = self.generate_game_data(games)
        if _tuple is not None:
            for x, y in zip(self.dataset, _tuple):
                x.extend(y)

    def collect_all_loaded_data(self):
        state_ary, policy_ary, value_ary = self.dataset

        state_ary1 = np.asarray(state_ary, dtype=np.float32)
        policy_ary1 = np.asarray(policy_ary, dtype=np.float32)
        value_ary1 = np.asarray(value_ary, dtype=np.float32)
        return state_ary1, policy_ary1, value_ary1

    def load_model(self):
        model = CChessModel(self.config)
        if self.config.opts.new or not load_sl_best_model_weight(model):
            model.build()
            save_as_sl_best_model(model)
        return model

    def save_current_model(self):
        logger.debug("Save best sl model")
        save_as_sl_best_model(self.model)

    def generate_game_data(self, games):
        self.buffer = []
        start_time = time()
        for idx, game in games.iterrows():
            gid = game['gameID']
            winner = game['winner']
            move = self.moves[self.moves.gameID == gid]
            red = move[move.side == 'red']
            black = move[move.side == 'black']
            self.load_game(red, black, winner, idx)
        end_time = time()
        logger.debug(f"Loading {len(games)} games, time: {end_time - start_time}s")
        return self.convert_to_trainging_data()

    def load_game(self, red, black, winner, idx):
        env = CChessEnv(self.config).reset()
        red_moves = []
        black_moves = []
        turns = 1
        black_max_turn = black['turn'].max()
        red_max_turn = red['turn'].max()

        while turns < black_max_turn or turns < red_max_turn:
            if turns < red_max_turn:
                wxf_move = red[red.turn == turns]['move'].item()
                action = env.board.parse_WXF_move(wxf_move)
                try:
                    red_moves.append([env.observation, self.build_policy(action, flip=False)])
                except Exception as e:
                    for i in range(10):
                        logger.debug(f"{env.board.screen[i]}")
                    logger.debug(f"{turns} {wxf_move} {action}")
                
                env.step(action)
            if turns < black_max_turn:
                wxf_move = black[black.turn == turns]['move'].item()
                action = env.board.parse_WXF_move(wxf_move)
                try:
                    black_moves.append([env.observation, self.build_policy(action, flip=True)])
                except Exception as e:
                    for i in range(10):
                        logger.debug(f"{env.board.screen[i]}")
                    logger.debug(f"{turns} {wxf_move} {action}")
                
                env.step(action)
            turns += 1

        if winner == 'red':
            red_win = 1
        elif winner == 'black':
            red_win = -1
        else:
            red_win = 0

        for move in red_moves:
            move += [red_win]
        for move in black_moves:
            move += [-red_win]

        data = []
        for i in range(len(red_moves)):
            data.append(red_moves[i])
            if i < len(black_moves):
                data.append(black_moves[i])
        self.buffer += data

    def build_policy(self, action, flip):
        labels_n = len(ActionLabelsRed)
        move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
        policy = np.zeros(labels_n)

        policy[move_lookup[action]] = 1

        if flip:
            policy = flip_policy(policy)
        return policy

    def convert_to_trainging_data(self):
        data = self.buffer
        state_list = []
        policy_list = []
        value_list = []
        env = CChessEnv()

        for state_fen, policy, value in data:
            state_planes = env.fen_to_planes(state_fen)
            sl_value = value

            state_list.append(state_planes)
            policy_list.append(policy)
            value_list.append(sl_value)

        return np.asarray(state_list, dtype=np.float32), \
               np.asarray(policy_list, dtype=np.float32), \
               np.asarray(value_list, dtype=np.float32)



