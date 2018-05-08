import os
import subprocess
import numpy as np
from collections import defaultdict
from logging import getLogger
from time import sleep, time
from datetime import datetime

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.environment.chessboard import Chessboard
from cchess_alphazero.environment.chessman import *
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_move
from cchess_alphazero.lib.model_helper import load_best_model_weight
from cchess_alphazero.lib.tf_util import set_session_config

logger = getLogger(__name__)

def start(config: Config, ucci=False, ai_move_first=True):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    if not ucci:
        play = ObSelfPlay(config)
    else:
        play = ObSelfPlayUCCI(config, ai_move_first)
    play.start()

class ObSelfPlay:
    def __init__(self, config: Config):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.mcts_pipe = None
        self.td_pipe = None
        self.mcts_ai = None
        self.td_ai = None
        self.chessmans = None

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def start(self):
        self.env.reset()
        self.load_model()
        self.mcts_pipe = self.model.get_pipes()
        self.td_pipe = self.model.get_pipes()

        from cchess_alphazero.agent.player import CChessPlayer
        from cchess_alphazero.agent.player import VisitState
        self.mcts_ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.mcts_pipe,
                              enable_resign=False, debugging=False)

        from cchess_alphazero.agent.td_player import CChessPlayer
        from cchess_alphazero.agent.td_player import VisitState
        self.td_ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.td_pipe,
                              enable_resign=False, debugging=False)

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        self.env.board.print_to_cl()
        history = [self.env.get_state()]

        while not self.env.board.is_end():
            no_act = None
            state = self.env.get_state()
            _, _, _, check = senv.done(state, need_check=True)
            if not check and state in history[:-1]:
                no_act = []
                free_move = defaultdict(int)
                for i in range(len(history) - 1):
                    if history[i] == state:
                        # 如果走了下一步是将军或捉：禁止走那步
                        if senv.will_check_or_catch(state, history[i+1]):
                            no_act.append(history[i + 1])
            # MCTS 执红，TD执黑
            if self.env.num_halfmoves % 2 == 0: 
                action, _ = self.mcts_ai.action(state, self.env.num_halfmoves, no_act)
            else:
                action, _ = self.td_ai.action(state, self.env.num_halfmoves, no_act)
            history.append(action)
            if action is None:
                print("AI投降了!")
                break
            move = self.env.board.make_single_record(int(action[0]), int(action[1]), int(action[2]), int(action[3]))
            if not self.env.red_to_move:
                action = flip_move(action)
            self.env.step(action)
            history.append(self.env.get_state())
            print(f"{move}")
            self.env.board.print_to_cl()

        self.mcts_ai.close()
        self.td_ai.close()
        print(f"胜者是 is {self.env.board.winner} !!!")
        logger.info(f"胜者是 is {self.env.board.winner} !!! MCTS 执红，TD执黑")
        self.env.board.print_record()
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(self.config.resource.play_record_dir, self.config.resource.play_record_filename_tmpl % game_id)
        self.env.board.save_record(path)
        logger.info(f"Save game record to {path}")

class ObSelfPlayUCCI:
    def __init__(self, config: Config, ai_move_first=True):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.chessmans = None
        self.ai_move_first = ai_move_first

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def start(self):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=False)

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        self.env.board.print_to_cl()
        history = [self.env.get_state()]
        turns = 0
        game_over = False
        final_move = None

        while not game_over:
            if (self.ai_move_first and turns % 2 == 0) or (not self.ai_move_first and turns % 2 == 1):
                start_time = time()
                no_act = None
                state = self.env.get_state()
                if state in history[:-1]:
                    no_act = []
                    for i in range(len(history) - 1):
                        if history[i] == state:
                            act = history[i + 1]
                            if not self.env.red_to_move:
                                act = flip_move(act)
                            no_act.append(act)
                action, _ = self.ai.action(state, self.env.num_halfmoves, no_act)
                end_time = time()
                if action is None:
                    print("AlphaZero 投降了!")
                    break
                move = self.env.board.make_single_record(int(action[0]), int(action[1]), int(action[2]), int(action[3]))
                print(f"AlphaZero 选择移动 {move}, 消耗时间 {(end_time - start_time):.2f}s")
                if not self.env.red_to_move:
                    action = flip_move(action)
            else:
                state = self.env.get_state()
                print(state)
                fen = senv.state_to_fen(state, turns)
                action = self.get_ucci_move(fen)
                if action is None:
                    print("Eleeye 投降了!")
                    break
                print(action)
                if not self.env.red_to_move:
                    rec_action = flip_move(action)
                else:
                    rec_action = action
                move = self.env.board.make_single_record(int(rec_action[0]), int(rec_action[1]), int(rec_action[2]), int(rec_action[3]))
                print(f"Eleeye 选择移动 {move}")
            history.append(action)
            self.env.step(action)
            history.append(self.env.get_state())
            self.env.board.print_to_cl()
            turns += 1
            sleep(1)
            game_over, final_move = self.env.board.is_end_final_move()
            print(game_over, final_move)

        if final_move:
            move = self.env.board.make_single_record(int(final_move[0]), int(final_move[1]), int(final_move[2]), int(final_move[3]))
            print(f"Final Move {move}")
            if not self.env.red_to_move:
                final_move = flip_move(final_move)
            self.env.step(final_move)
            self.env.board.print_to_cl()

        self.ai.close()
        print(f"胜者是 is {self.env.board.winner} !!!")
        self.env.board.print_record()

    def get_ucci_move(self, fen, time=3):
        p = subprocess.Popen(self.config.resource.eleeye_path,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True)
        setfen = f'position fen {fen}\n'
        setrandom = 'setoption randomness small\n'
        cmd = 'ucci\n' + setrandom + setfen + f'go time {time * 1000}\n'
        try:
            out, err = p.communicate(cmd, timeout=time+0.5)
        except:
            p.kill()
            try:
                out, err = p.communicate()
            except Exception as e:
                logger.error(f"{e}, cmd = {cmd}")
                return self.get_ucci_move(fen, time+1)
        print(out)
        lines = out.split('\n')
        if lines[-2] == 'nobestmove':
            return None
        move = lines[-2].split(' ')[1]
        if move == 'depth':
            move = lines[-1].split(' ')[6]
        return senv.parse_ucci_move(move)
