from multiprocessing import connection, Pipe
from threading import Thread

import numpy as np

from cchess_alphazero.config import Config
from cchess_alphazero.lib.model_helper import load_best_model_weight, need_to_reload_best_model_weight
from cchess_alphazero.lib.web_helper import http_request, download_file
from time import time
from logging import getLogger

logger = getLogger(__name__)

class CChessModelAPI:

    def __init__(self, config: Config, agent_model):  
        self.agent_model = agent_model  # CChessModel
        self.pipes = []     # use for communication between processes/threads
        self.config = config
        self.need_reload = True
        self.done = False

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
        self.try_reload_model_from_internet()
        last_model_check_time = time()
        while not self.done:
            if last_model_check_time + 600 < time():
                self.try_reload_model()
                last_model_check_time = time()
            ready = connection.wait(self.pipes, timeout=0.001)
            if not ready:
                continue
            data, result_pipes, data_len = [], [], []
            for pipe in ready:
                while pipe.poll():
                    try:
                        tmp = pipe.recv()
                        data.extend(tmp)
                        data_len.append(len(tmp))
                        result_pipes.append(pipe)
                    except EOFError as e:
                        logger.error(f"EOF error: {e}")
                        pipe.close()
            data = np.asarray(data, dtype=np.float32)
            with self.agent_model.graph.as_default():
                policy_ary, value_ary = self.agent_model.model.predict_on_batch(data)
            buf = []
            k, i = 0, 0
            for p, v in zip(policy_ary, value_ary):
                buf.append((p, float(v)))
                k += 1
                if k >= data_len[i]:
                    result_pipes[i].send(buf)
                    buf = []
                    k = 0
                    i += 1

    def try_reload_model(self):
        try:
            if self.config.internet.distributed:
                # reload_worker = Thread(target=self.try_reload_model_from_internet, name="reload_worker")
                # reload_worker.daemon = True
                # reload_worker.start()
                self.try_reload_model_from_internet()
            else:
                if self.need_reload and need_to_reload_best_model_weight(self.agent_model):
                    with self.agent_model.graph.as_default():
                        load_best_model_weight(self.agent_model)
        except Exception as e:
            logger.error(e)

    def try_reload_model_from_internet(self):
        response = http_request(self.config.internet.get_latest_digest)
        if response is None:
            logger.error(f"Could not fetch remote digest!")
            return
        digest = response['data']['digest']

        if digest != self.agent_model.fetch_digest(self.config.resource.model_best_weight_path):
            logger.info("the best model is changed, start download remote model")
            if download_file(self.config.internet.download_url, self.config.resource.model_best_weight_path):
                logger.info(f"Download remote model finished!")
                with self.agent_model.graph.as_default():
                    load_best_model_weight(self.agent_model)
            else:
                logger.error(f"Download remote model failed!")
        else:
            logger.info(f'the best model is not changed')

    def close(self):
        self.done = True
