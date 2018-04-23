from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock, Condition
import concurrent.futures.thread

import numpy as np
import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_move
from time import time, sleep
import gc 
import sys

logger = getLogger(__name__)

class VisitState:
    def __init__(self):
        self.a = defaultdict(ActionState)   # key: action, value: ActionState
        self.sum_n = 0                      # visit count
        self.visit = []                     # thread id that has visited this state
        self.p = None                       # policy of this state
        self.legal_moves = None             # all leagal moves of this state
        self.waiting = False                # is waiting for NN's predict
        self.w = 0


class ActionState:
    def __init__(self):
        self.n = 0      # N(s, a) : visit count
        self.w = 0      # W(s, a) : total action value
        self.q = 0      # Q(s, a) = N / W : action value
        self.p = 0      # P(s, a) : prior probability

class CChessPlayer:
    def __init__(self, config: Config, search_tree=None, pipes=None, play_config=None, enable_resign=False, debugging=False, uci=False):
        self.config = config
        self.play_config = play_config or self.config.play
        self.labels_n = len(ActionLabelsRed)
        self.labels = ActionLabelsRed
        self.move_lookup = {move: i for move, i in zip(self.labels, range(self.labels_n))}
        self.pipe = pipes                   # pipes that used to communicate with CChessModelAPI thread
        self.node_lock = defaultdict(Lock)  # key: state key, value: Lock of that state

        if search_tree is None:
            self.tree = defaultdict(VisitState)  # key: state key, value: VisitState
        else:
            self.tree = search_tree

        self.root_state = None

        self.enable_resign = enable_resign
        self.debugging = debugging

        self.search_results = {}        # for debug
        self.debug = {}

        self.s_lock = Lock()
        self.run_lock = Lock()
        self.q_lock = Lock()            # queue lock
        self.t_lock = Lock()
        self.buffer_planes = []         # prediction queue
        self.buffer_history = []

        self.all_done = Lock()
        self.num_task = 0
        self.done_tasks = 0
        self.uci = uci

        self.job_done = False

        self.executor = ThreadPoolExecutor(max_workers=self.play_config.search_threads + 2)
        self.executor.submit(self.receiver)
        self.executor.submit(self.sender)

    def close(self, wait=True):
        self.job_done = True
        del self.tree
        gc.collect()
        if self.executor is not None:
            self.executor.shutdown(wait=wait)

    def close_and_return_action(self, state, turns, no_act=None):
        self.job_done = True
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            # self.executor = None
            self.executor._threads.clear()
            concurrent.futures.thread._threads_queues.clear()
        policy, resign = self.calc_policy(state, turns)
        if resign:  # resign
            return None
        if no_act is not None:
            for act in no_act:
                policy[self.move_lookup[act]] = 0
        my_action = int(np.random.choice(range(self.labels_n), p=self.apply_temperature(policy, turns)))
        if state in self.debug:
            _, value = self.debug[state]
        else:
            value = 0
        return self.labels[my_action], value

    def sender(self):
        '''
        send planes to neural network for prediction
        '''
        limit = 256                 # max prediction queue size
        while not self.job_done:
            self.run_lock.acquire()
            with self.q_lock:
                l = min(limit, len(self.buffer_history))
                if l > 0:
                    t_data = self.buffer_planes[0:l]
                    # logger.debug(f"send queue size = {l}")
                    self.pipe.send(t_data)
                else:
                    self.run_lock.release()
                    sleep(0.001)

    def receiver(self):
        '''
        receive policy and value from neural network
        '''
        while not self.job_done:
            if self.pipe.poll(0.001):
                rets = self.pipe.recv()
            else:
                continue
            k = 0
            with self.q_lock:
                for ret in rets:
                    # logger.debug(f"NN ret, update tree buffer_history = {self.buffer_history}")
                    self.executor.submit(self.update_tree, ret[0], ret[1], self.buffer_history[k])
                    # self.update_tree(ret[0], ret[1], self.buffer_history[k])
                    k = k + 1
                self.buffer_planes = self.buffer_planes[k:]
                self.buffer_history = self.buffer_history[k:]
            self.run_lock.release()

    def action(self, state, turns, no_act=None, depth=None, infinite=False) -> str:
        self.all_done.acquire(True)
        self.root_state = state
        done = 0
        if state in self.tree:
            done = self.tree[state].sum_n
        self.done_tasks = done
        self.num_task = self.play_config.simulation_num_per_move - done
        if depth:
            self.num_task = depth - done if depth > done else 0
        if infinite:
            self.num_task = 100000
        depth = 0
        start_time = time()
        # MCTS search
        if self.num_task > 0:
            all_tasks = self.num_task
            batch = all_tasks // self.config.play.search_threads
            if all_tasks % self.config.play.search_threads != 0:
                batch += 1
            # logger.debug(f"all_task = {self.num_task}, batch = {batch}")
            for iter in range(batch):
                self.num_task = min(self.config.play.search_threads, all_tasks - self.config.play.search_threads * iter)
                self.done_tasks += self.num_task
                # logger.debug(f"iter = {iter}, num_task = {self.num_task}")
                for i in range(self.num_task):
                    self.executor.submit(self.MCTS_search, state, [state], True)
                self.all_done.acquire(True)
                if self.uci and depth != self.done_tasks // 100:
                    # info depth xx pv xxx
                    depth = self.done_tasks // 100
                    _, value = self.debug[state]
                    self.print_depth_info(state, turns, start_time, value)
        self.all_done.release()

        policy, resign = self.calc_policy(state, turns)

        if resign:  # resign
            return None, list(policy)
        if no_act is not None:
            for act in no_act:
                policy[self.move_lookup[act]] = 0

        my_action = int(np.random.choice(range(self.labels_n), p=self.apply_temperature(policy, turns)))
        return self.labels[my_action], list(policy)

    def MCTS_search(self, state, history=[], is_root_node=False) -> float:
        """
        Monte Carlo Tree Search
        """
        while True:
            # logger.debug(f"start MCTS, state = {state}, history = {history}")
            game_over, v, _ = senv.done(state)
            if game_over:
                self.executor.submit(self.update_tree, None, v, history)
                break

            with self.node_lock[state]:
                if state not in self.tree:
                    # Expand and Evaluate
                    self.tree[state].sum_n = 1
                    self.tree[state].legal_moves = senv.get_legal_moves(state)
                    self.tree[state].waiting = True
                    # logger.debug(f"expand_and_evaluate {state}, sum_n = {self.tree[state].sum_n}, history = {history}")
                    self.expand_and_evaluate(state, history)
                    break

                if state in history[:-1]: # loop -> loss
                    # logger.debug(f"loop -> loss, state = {state}, history = {history[:-1]}")
                    self.executor.submit(self.update_tree, None, 0, history)
                    break

                # Select
                node = self.tree[state]
                if node.waiting:
                    node.visit.append(history)
                    # logger.debug(f"wait for prediction state = {state}")
                    break

                sel_action = self.select_action_q_and_u(state, is_root_node)

                virtual_loss = self.config.play.virtual_loss
                self.tree[state].sum_n += 1
                # logger.debug(f"node = {state}, sum_n = {node.sum_n}")
                
                action_state = self.tree[state].a[sel_action]
                action_state.n += virtual_loss
                action_state.w -= virtual_loss
                action_state.q = action_state.w / action_state.n

                # logger.debug(f"apply virtual_loss = {virtual_loss}, as.n = {action_state.n}, w = {action_state.w}, q = {action_state.q}")
                
                # if action_state.next is None:
                history.append(sel_action)
                state = senv.step(state, sel_action)
                history.append(state)
                # logger.debug(f"step action {sel_action}, next = {action_state.next}")

            # history.append(sel_action)
            # state = action_state.next
            # history.append(state)

    def select_action_q_and_u(self, state, is_root_node) -> str:
        '''
        Select an action with highest Q(s,a) + U(s,a)
        '''
        is_root_node = self.root_state == state
        # logger.debug(f"select_action_q_and_u for {state}, root = {is_root_node}")
        node = self.tree[state]
        legal_moves = node.legal_moves

        # push p, the prior probability to the edge (node.p), only consider legal moves
        if node.p is not None:
            all_p = 0
            for mov in legal_moves:
                mov_p = node.p[self.move_lookup[mov]]
                node.a[mov].p = mov_p
                all_p += mov_p
            # rearrange the distribution
            if all_p == 0:
                all_p = 1
            for mov in legal_moves:
                node.a[mov].p /= all_p
            # release the temp policy
            node.p = None

        # sqrt of sum(N(s, b); for all b)
        xx_ = np.sqrt(node.sum_n + 1)  

        e = self.play_config.noise_eps
        c_puct = self.play_config.c_puct
        dir_alpha = self.play_config.dirichlet_alpha

        best_score = -99999999
        best_action = None

        for mov in legal_moves:
            action_state = node.a[mov]
            p_ = action_state.p
            if is_root_node:
                p_ = (1 - e) * p_ + e * np.random.dirichlet([dir_alpha])[0]
            # Q + U
            score = action_state.q + c_puct * p_ * xx_ / (1 + action_state.n)
            # if score > 0.1 and is_root_node:
            #     logger.debug(f"U+Q = {score:.2f}, move = {mov}, q = {action_state.q:.2f}")
            if action_state.q > (1 - 1e-7):
                best_action = mov
                break
            if score > best_score:
                best_score = score
                best_action = mov

        if best_action == None:
            logger.error(f"Best action is None, legal_moves = {legal_moves}, best_score = {best_score}")
        # if is_root_node:
        #     logger.debug(f"selected action = {best_action}, with U + Q = {best_score}")
        return best_action

    def expand_and_evaluate(self, state, history):
        '''
        Evaluate the state, return its policy and value computed by neural network
        '''
        state_planes = senv.state_to_planes(state)
        with self.q_lock:
            self.buffer_planes.append(state_planes)
            self.buffer_history.append(history)
            # logger.debug(f"EAE append buffer_history history = {history}")

    def update_tree(self, p, v, history):
        state = history.pop()
        z = v

        if p is not None:
            with self.node_lock[state]:
                # logger.debug(f"return from NN state = {state}, v = {v}")
                node = self.tree[state]
                node.p = p
                node.waiting = False
                if self.debugging:
                    self.debug[state] = (p, v)
                for hist in node.visit:
                    self.executor.submit(self.MCTS_search, state, hist)
                node.visit = []

        virtual_loss = self.config.play.virtual_loss
        # logger.debug(f"backup from {state}, v = {v}, history = {history}")
        while len(history) > 0:
            action = history.pop()
            state = history.pop()
            v = - v
            with self.node_lock[state]:
                node = self.tree[state]
                action_state = node.a[action]
                action_state.n += 1 - virtual_loss
                action_state.w += v + virtual_loss
                action_state.q = action_state.w * 1.0 / action_state.n
                # logger.debug(f"update value: state = {state}, action = {action}, n = {action_state.n}, w = {action_state.w}, q = {action_state.q}")

        with self.t_lock:
            self.num_task -= 1
            # logger.debug(f"finish 1, remain num task = {self.num_task}")
            if self.num_task <= 0:
                self.all_done.release()

    def calc_policy(self, state, turns) -> np.ndarray:
        '''
        calculate π(a|s0) according to the visit count
        '''
        node = self.tree[state]
        policy = np.zeros(self.labels_n)
        max_q_value = -100
        debug_result = {}

        for mov, action_state in node.a.items():
            policy[self.move_lookup[mov]] = action_state.n
            if self.debugging:
                debug_result[mov] = (action_state.n, action_state.q, action_state.p)
            if action_state.q > max_q_value:
                max_q_value = action_state.q

        if max_q_value < self.play_config.resign_threshold and self.enable_resign and turns > self.play_config.min_resign_turn:
            return policy, True

        if self.debugging:
            temp = sorted(range(len(policy)), key=lambda k: policy[k], reverse=True)
            for i in range(5):
                index = temp[i]
                mov = ActionLabelsRed[index]
                if mov in debug_result:
                    self.search_results[mov] = debug_result[mov]

        policy /= np.sum(policy)
        return policy, False

    def print_depth_info(self, state, turns, start_time, value):
        '''
        info depth xx pv xxx
        '''
        depth = self.done_tasks // 100
        end_time = time()
        score = int(value * 1000)
        output = f"info depth {depth} score {score} time {int((end_time - start_time) * 1000)} pv"
        i = 0
        while i < 10:
            node = self.tree[state]
            bestmove = None
            n = 0
            if len(node.a) == 0:
                break
            for mov, action_state in node.a.items():
                if action_state.n > n:
                    n = action_state.n
                    bestmove = mov
            state = senv.step(state, bestmove)
            if turns % 2 == 1:
                bestmove = flip_move(bestmove)
            bestmove = senv.to_uci_move(bestmove)
            output += " " + bestmove
            i += 1
            turns += 1
        print(output)
        logger.debug(output)
        sys.stdout.flush()
        

    def apply_temperature(self, policy, turn) -> np.ndarray:
        if turn < 30 and self.play_config.tau_decay_rate != 0:
            tau = tau = np.power(self.play_config.tau_decay_rate, turn + 1)
        else:
            tau = 0
        if tau < 0.1:
             tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.labels_n)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1 / tau)
            ret /= np.sum(ret)
            return ret


