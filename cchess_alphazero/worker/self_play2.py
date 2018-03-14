import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
from threading import Thread
from time import time
from collections import defaultdict
from threading import Lock

from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner
from cchess_alphazero.lib.data_helper import get_game_data_filenames, write_game_data_to_file
from cchess_alphazero.lib.model_helper import load_best_model_weight, save_as_best_model

logger = getLogger(__name__)
job_done = Lock()
thr_free = Lock()
env = None
data = None
futures = []

def start(config: Config):
    return SelfPlayWorker(config).start()

class SelfPlayWorker:
    def __init__(self, config: Config):
        self.config = config
        self.current_model = self.load_model()
        self.m = Manager()
        self.cur_pipes = self.m.list([self.current_model.get_pipes(self.config.play.search_threads) \
                        for _ in range(self.config.play.max_processes)])

    def start(self):
        global job_done
        global thr_free
        global env
        global data
        global futures

        self.buffer = []
        job_done.acquire(True)

        futures = []
        with ProcessPoolExecutor(max_workers=self.config.play.max_processes) as executor:
            game_idx = 0
            while True:
                game_idx += 1
                start_time = time()

                if len(futures) == 0:
                    for i in range(self.config.play.max_processes):
                        ff = executor.submit(self_play_buffer, self.config, cur=self.cur_pipes)
                        ff.add_done_callback(recall_fn)
                        futures.append(ff)

                job_done.acquire(True)

                end_time = time()
                logger.debug(f"Play game {game_idx} time={end_time - start_time} sec, "
                             f"turn={env.num_halfmoves / 2}:{env.winner}")

                self.buffer += data

                if (game_idx % self.config.play_data.nb_game_in_file) == 0:
                    self.flush_buffer()
                    self.remove_play_data(all=False) # remove old data
                if not need_to_renew_model: # avoid congestion
                    ff = executor.submit(self_play_buffer, self.config, cur=self.cur_pipes)
                    ff.add_done_callback(recall_fn)
                    futures.append(ff) # Keep it going
                thr_free.release()

        if len(data) > 0:
            self.flush_buffer()

    def load_model(self):
        model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(model):
            model.build()
            save_as_best_model(model)
        return model

    def flush_buffer(self):
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info("save play data to %s" % (path))
        thread = Thread(target=write_game_data_to_file, args=(path, self.buffer))
        thread.start()
        self.buffer = []

    def remove_play_data(self,all=False):
        files = get_game_data_filenames(self.config.resource)
        if (all):
            for path in files:
                os.remove(path)
        else:
            while len(files) > self.config.play_data.max_file_num:
                os.remove(files[0])
                del files[0]

def recall_fn(future):
    global thr_free
    global job_done
    global env
    global data
    global futures

    thr_free.acquire(True)
    env, data = future.result()
    futures.remove(future)
    job_done.release()

def self_play_buffer(config, cur) -> (CChessEnv, list):
    pipes = cur.pop() # borrow
    env = CChessEnv(config).reset()
    search_tree = defaultdict(VisitState)

    red = CChessPlayer(config, search_tree=search_tree, pipes=pipes)
    black = CChessPlayer(config, search_tree=search_tree, pipes=pipes)

    history = []

    cc = 0
    while not env.done:
        start_time = time()
        if env.red_to_move:
            action = red.action(env)
        else:
            action = black.action(env)
        end_time = time()
        logger.debug(f"Playing: {env.red_to_move}, action: {action}, time: {end_time - start_time}s")
        env.step(action)
        history.append(action)
        if len(history) > 6 and history[-1] == history[-5]:
            cc = cc + 1
        else:
            cc = 0
        if env.num_halfmoves / 2 >= config.play.max_game_length:
                env.winner = Winner.draw
        if cc >= 4:
            if env.red_to_move:
                env.winner = Winner.black
            else:
                env.winner = Winner.red
    if env.winner == Winner.red:
        black_win = -1
    elif env.winner == Winner.black:
        black_win = 1
    else:
        black_win = 0

    black.finish_game(black_win)
    red.finish_game(-black_win)

    data = []
    for i in range(len(red.moves)):
        data.append(red.moves[i])
        if i < len(black.moves):
            data.append(black.moves[i])

    cur.append(pipes)
    return env, data
