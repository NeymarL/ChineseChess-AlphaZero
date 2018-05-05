import os
import time
import subprocess
import shutil
import numpy as np

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from time import sleep
from random import shuffle
from threading import Thread

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.config import Config
from cchess_alphazero.lib.data_helper import get_game_data_filenames, read_game_data_from_file
from cchess_alphazero.lib.model_helper import load_best_model_weight, save_as_best_model
from cchess_alphazero.lib.model_helper import need_to_reload_best_model_weight, save_as_next_generation_model, save_as_best_model
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_policy, flip_move
from cchess_alphazero.lib.tf_util import set_session_config
from cchess_alphazero.lib.web_helper import http_request

from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.utils import multi_gpu_model
import keras.backend as K

logger = getLogger(__name__)

def start(config: Config):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    return OptimizeWorker(config).start()

class OptimizeWorker:
    def __init__(self, config:Config):
        self.config = config
        self.model = None
        self.loaded_filenames = set()
        self.loaded_data = deque(maxlen=self.config.trainer.dataset_size)
        self.dataset = deque(), deque(), deque()
        self.executor = ProcessPoolExecutor(max_workers=config.trainer.cleaning_processes)
        self.filenames = []
        self.opt = None
        self.count = 0
        self.eva = False

    def start(self):
        self.model = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        total_steps = self.config.trainer.start_total_steps
        bef_files = []
        last_file = None

        while True:
            files = get_game_data_filenames(self.config.resource)
            offset = self.config.trainer.min_games_to_begin_learn // self.config.play_data.nb_game_in_file
            if (len(files) * self.config.play_data.nb_game_in_file < self.config.trainer.min_games_to_begin_learn \
              or ((last_file is not None) and files.index(last_file) + offset > len(files))):
                if last_file is not None:
                    logger.info('Waiting for enough data 300s, ' + str((len(files) - files.index(last_file)) * self.config.play_data.nb_game_in_file) \
                            +' vs '+ str(self.config.trainer.min_games_to_begin_learn)+' games')
                else:
                    logger.info('Waiting for enough data 300s, ' + str(len(files) * self.config.play_data.nb_game_in_file) \
                            +' vs '+ str(self.config.trainer.min_games_to_begin_learn)+' games')
                time.sleep(300)
                continue
            else:
                bef_files = files
                if last_file is not None and len(files) > offset:
                    files = files[-offset:]
                last_file = files[-1]
                logger.info(f"Last file = {last_file}")
                self.filenames = deque(files)
                logger.debug(f"Start training {len(self.filenames)} files")
                shuffle(self.filenames)
                self.fill_queue()
                if len(self.dataset[0]) > self.config.trainer.batch_size:
                    steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
                    total_steps += steps
                    self.save_current_model()
                    self.update_learning_rate(total_steps)
                    self.count += 1
                    a, b, c = self.dataset
                    a.clear()
                    b.clear()
                    c.clear()
                    self.remove_play_data()
                    break

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, value_ary = self.collect_all_loaded_data()
        tensorboard_cb = TensorBoard(log_dir="./logs", batch_size=tc.batch_size, histogram_freq=1)
        self.model.model.fit(state_ary, [policy_ary, value_ary],
                             batch_size=tc.batch_size,
                             epochs=epochs,
                             shuffle=True,
                             validation_split=0.02,
                             callbacks=[tensorboard_cb])
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        self.opt = SGD(lr=0.01, momentum=self.config.trainer.momentum)
        losses = ['categorical_crossentropy', 'mean_squared_error'] # avoid overfit for supervised 
        if self.config.opts.use_multiple_gpus:
            model = multi_gpu_model(self.model.model, cpu_merge=False, gpus=self.config.opts.gpu_num)
        else:
            model = self.model.model
        model.compile(optimizer=self.opt, loss=losses, loss_weights=self.config.trainer.loss_weights)

    def update_learning_rate(self, total_steps):
        # The deepmind paper says
        # ~400k: 1e-2
        # 400k~600k: 1e-3
        # 600k~: 1e-4

        lr = self.decide_learning_rate(total_steps)
        if lr:
            K.set_value(self.opt.lr, lr)
            logger.debug(f"total step={total_steps}, set learning rate to {lr}")

    def fill_queue(self):
        futures = deque()
        with ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes) as executor:
            for _ in range(self.config.trainer.cleaning_processes):
                if len(self.filenames) == 0:
                    break
                filename = self.filenames.pop()
                logger.debug("loading data from %s" % (filename))
                futures.append(executor.submit(load_data_from_file, filename))
            while futures and len(self.dataset[0]) < self.config.trainer.dataset_size: #fill tuples
                _tuple = futures.popleft().result()
                if _tuple is not None:
                    for x, y in zip(self.dataset, _tuple):
                        x.extend(y)
                if len(self.filenames) > 0:
                    filename = self.filenames.pop()
                    logger.debug("loading data from %s" % (filename))
                    futures.append(executor.submit(load_data_from_file, filename))

    def collect_all_loaded_data(self):
        state_ary, policy_ary, value_ary = self.dataset

        state_ary1 = np.asarray(state_ary, dtype=np.float32)
        policy_ary1 = np.asarray(policy_ary, dtype=np.float32)
        value_ary1 = np.asarray(value_ary, dtype=np.float32)
        return state_ary1, policy_ary1, value_ary1

    def load_model(self):
        model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(model):
            model.build()
            save_as_best_model(model)
        return model

    def save_current_model(self):
        logger.info("Save as ng model")
        # save_as_best_model(self.model)
        save_as_next_generation_model(self.model)
        # -------------- debug --------------
        if self.count % 1 == 0:
            self.eva = True
        else:
            self.eva = False
        if self.config.internet.distributed:
            # send_worker = Thread(target=self.send_model, name="send_worker")
            # send_worker.daemon = True
            # send_worker.start()
            self.send_model()

    def decide_learning_rate(self, total_steps):
        ret = None

        for step, lr in self.config.trainer.lr_schedules:
            if total_steps >= step:
                ret = lr
        return ret

    def try_reload_model(self):
        logger.debug("check model")
        if need_to_reload_best_model_weight(self.model):
            with self.model.graph.as_default():
                load_best_model_weight(self.model)
            return True
        return False

    def send_model(self):
        success = False
        remote_server = 'root@111.231.100.42'
        # for i in range(3):
        #     remote_server = 'root@111.231.100.42'
        #     remote_path = '/var/www/alphazero.52coding.com.cn/data/model/128x7/model_best_weight.h5'
        #     cmd = f'scp {self.config.resource.next_generation_weight_path} {remote_server}:{remote_path}'
        #     ret = subprocess.run(cmd, shell=True)
        #     if ret.returncode == 0:
        #         success = True
        #         logger.info("Send best model success!")
        #         break
        #     else:
        #         logger.error(f"Send best model failed! {ret.stderr}, cmd = {cmd}")
        # if self.eva:
        filename = self.model.digest + '.h5'
        weight_path = os.path.join(self.config.resource.next_generation_model_dir, filename)
        shutil.copy(self.config.resource.next_generation_weight_path, weight_path)
        for i in range(3):
            remote_path = '/var/www/alphazero.52coding.com.cn/data/model/next_generation'
            cmd = f'scp {weight_path} {remote_server}:{remote_path}'
            ret = subprocess.run(cmd, shell=True)
            if ret.returncode == 0:
                success = True
                logger.info("Send evaluate model success!")
                os.remove(weight_path)
                break
            else:
                logger.error(f"Send evaluate model failed! {ret.stderr}, cmd = {cmd}")
        data = {'digest': self.model.digest, 'elo': 0}
        http_request(self.config.internet.add_model_url, post=True, data=data)

    def remove_play_data(self):
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.play_data.max_file_num:
            return
        backup_folder = os.path.join(self.config.resource.data_dir, 'backup');
        try:
            for i in range(len(files) - self.config.play_data.max_file_num):
                # os.remove(files[i])
                shutil.move(files[i], backup_folder)
        except:
            pass

def load_data_from_file(filename):
    try:
        data = read_game_data_from_file(filename)
    except Exception as e:
        logger.error(f"Error when loading data {e}")
        return None
    if data is None:
        return None
    return expanding_data(data)

def expanding_data(data):
    state = data[0]
    real_data = []
    action = None
    policy = None
    value = None
    for item in data[1:]:
        action = item[0]
        value = item[1]
        try:
            policy = build_policy(action, flip=False)
        except Exception as e:
            logger.error(f"Expand data error {e}, item = {item}, data = {data}, state = {state}")
            return None
        real_data.append([state, policy, value])
        state = senv.step(state, action)
        
    return convert_to_trainging_data(real_data)


def convert_to_trainging_data(data):
    state_list = []
    policy_list = []
    value_list = []

    for state, policy, value in data:
        state_planes = senv.state_to_planes(state)
        sl_value = value

        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(sl_value)

    return np.asarray(state_list, dtype=np.float32), \
           np.asarray(policy_list, dtype=np.float32), \
           np.asarray(value_list, dtype=np.float32)

def build_policy(action, flip):
    labels_n = len(ActionLabelsRed)
    move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
    policy = np.zeros(labels_n)

    policy[move_lookup[action]] = 1

    if flip:
        policy = flip_policy(policy)
    return list(policy)



