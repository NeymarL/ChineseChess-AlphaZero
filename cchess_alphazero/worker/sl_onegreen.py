import os
import numpy as np
import json

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from time import sleep
from random import shuffle
from time import time

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.config import Config
from cchess_alphazero.lib.data_helper import get_game_data_filenames, read_game_data_from_file
from cchess_alphazero.lib.model_helper import load_sl_best_model_weight, save_as_sl_best_model
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, flip_policy, flip_move
from cchess_alphazero.lib.tf_util import set_session_config
from cchess_alphazero.environment.lookup_tables import Winner

from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import keras.backend as K

logger = getLogger(__name__)

def start(config: Config, skip):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    return SupervisedWorker(config).start(skip)

class SupervisedWorker:
    def __init__(self, config:Config):
        self.config = config
        self.model = None
        self.loaded_data = deque(maxlen=self.config.trainer.dataset_size)
        self.dataset = deque(), deque(), deque()
        self.filenames = []
        self.opt = None
        self.buffer = []
        self.games = None

    def start(self, skip=0):
        self.model = self.load_model()
        with open(self.config.resource.sl_onegreen, 'r') as f:
            self.games = json.load(f)
        self.training(skip)

    def training(self, skip=0):
        self.compile_model()
        total_steps = self.config.trainer.start_total_steps
        logger.info(f"Start training, game count = {len(self.games)}, step = {self.config.trainer.sl_game_step} games, skip = {skip}")

        for i in range(skip, len(self.games), self.config.trainer.sl_game_step):
            games = self.games[i:i+self.config.trainer.sl_game_step]
            self.fill_queue(games)
            if len(self.dataset[0]) > self.config.trainer.batch_size:
                steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
                total_steps += steps
                self.save_current_model()
                a, b, c = self.dataset
                a.clear()
                b.clear()
                c.clear()
                logger.debug(f"total steps = {total_steps}")

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
        self.opt = Adam(lr=0.003)
        losses = ['categorical_crossentropy', 'mean_squared_error']
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
        idx = 0
        cnt = 0
        for game in games:
            init = game['init']
            move_list = game['move_list']
            winner = Winner.draw
            if game['result'] == '红胜' or '胜' in game['title']:
                winner = Winner.red
            elif game['result'] == '黑胜' or '负' in game['title']:
                winner = Winner.black
            else:
                winner = Winner.draw
            v = self.load_game(init, move_list, winner, idx)
            if v == 1 or v == -1:
                cnt += 1
            idx += 1
        end_time = time()
        logger.debug(f"Loading {len(games)} games, time: {end_time - start_time}s, end games = {cnt}")
        return self.convert_to_trainging_data()

    def load_game(self, init, move_list, winner, idx):
        turns = 0
        if init == '':
            state = senv.INIT_STATE
        else:
            state = senv.init(init)
        moves = [move_list[i:i+4] for i in range(len(move_list)) if i % 4 == 0]
        history = []
        policys = []

        for move in moves:
            action = senv.parse_onegreen_move(move)
            if turns % 2 == 1:
                action = flip_move(action)                
            try:
                policy = self.build_policy(action, False)
            except:
                logger.error(f"idx = {idx}, action = {action}, turns = {turns}, moves = {moves}, winner = {winner}, init = {init}")
                return

            history.append(action)
            policys.append(policy)

            state = senv.step(state, action)
            turns += 1

        if winner == Winner.red:
            value = 1
        elif winner == Winner.black:
            value = -1
        else:
            game_over, value = senv.done(state)
            if not game_over:
                value = senv.evaluate(state)
            if turns % 2 == 1:  # balck turn
                value = -value

        data = []
        for i in range(turns):
            data.append([history[i], policys[i], value])
            value = -value
        self.buffer += data
        return value

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



