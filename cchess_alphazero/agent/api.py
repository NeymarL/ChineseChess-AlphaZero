from multiprocessing import connection, Pipe
from threading import Thread

import numpy as np

from cchess_alphazero.config import Config
from cchess_alphazero.lib.model_helper import load_best_model_weight, need_to_reload_best_model_weight
from time import time
from logging import getLogger

logger = getLogger(__name__)

class CChessModelAPI:

    def __init__(self, config: Config, agent_model):  
        self.agent_model = agent_model  # CChessModel
        self.pipes = []     # use for communication between processes/threads
        self.config = config
        self.need_reload = True

    def start(self):
        prediction_worker = Thread(target=self.predict_batch_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def get_pipe(self, need_reload=True):
        me, you = Pipe()
        self.pipes.append(me)
        self.need_reload = need_reload
        return you

    def predict_batch_worker(self):
        last_model_check_time = time()
        while True:
            if last_model_check_time + 600 < time():
                self.try_reload_model()
                last_model_check_time = time()
            ready = connection.wait(self.pipes, timeout=0.001)
            if not ready:
                continue
            data, result_pipes = [], []
            for pipe in ready:
                while pipe.poll():
                    try:
                        data.append(pipe.recv())
                        result_pipes.append(pipe)
                    except EOFError as e:
                        pipe.close()
            data = np.asarray(data, dtype=np.float32)
            with self.agent_model.graph.as_default():
                policy_ary, value_ary = self.agent_model.model.predict_on_batch(data)
            for pipe, p, v in zip(result_pipes, policy_ary, value_ary):
                pipe.send((p, float(v)))

    def try_reload_model(self):
        try:
            logger.debug("check model")
            if not self.config.opts.evaluate and need_to_reload_best_model_weight(self.agent_model) and need_reload:
                with self.agent_model.graph.as_default():
                    load_best_model_weight(self.agent_model)
        except Exception as e:
            logger.error(e)
