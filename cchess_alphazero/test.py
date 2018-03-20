import os
import sys
import multiprocessing as mp

_PATH_ = os.path.dirname(os.path.dirname(__file__))


if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

def test_env():
    from cchess_alphazero.environment.env import CChessEnv
    env = CChessEnv()
    env.reset()
    print(env.observation)
    env.step('0001')
    print(env.observation)
    env.step('7770')
    print(env.observation)
    env.render()
    print(env.input_planes()[0+7:3+7])

def test_player():
    from cchess_alphazero.agent.player import CChessPlayer

def test_config():
    from cchess_alphazero.config import Config
    c = Config('mini')
    c.resource.create_directories()
    print(c.resource.project_dir, c.resource.data_dir)

def test_self_play():
    from cchess_alphazero.config import Config
    from cchess_alphazero.worker.self_play import start
    from cchess_alphazero.lib.logger import setup_logger
    c = Config('mini')
    c.resource.create_directories()
    setup_logger(c.resource.main_log_path)
    start(c)

def test_cli_play():
    from cchess_alphazero.play_games.test_cli_game import main
    main()

def test_gui_play():
    from cchess_alphazero.play_games.test_window_game import main
    main()

def test_optimise():
    from cchess_alphazero.worker.optimize import start
    from cchess_alphazero.config import Config
    from cchess_alphazero.lib.logger import setup_logger
    c = Config('mini')
    c.resource.create_directories()
    setup_logger(c.resource.main_log_path)
    start(c)

def test_light():
    from cchess_alphazero.environment.light_env.chessboard import L_Chessboard
    from cchess_alphazero.environment.chessboard import Chessboard
    lboard = L_Chessboard()
    while not lboard.is_end():
        for i in range(lboard.height):
            print(lboard.screen[i])
        print(f"legal_moves = {lboard.legal_moves()}")
        action = input(f'Enter move for {lboard.is_red_turn} r/b: ')
        lboard.move_action_str(action)
    for i in range(lboard.height):
        print(lboard.screen[i])
    print(lboard.winner)
    print(f"Turns = {lboard.steps / 2}")

def test_light_env():
    from cchess_alphazero.environment.env import CChessEnv
    from cchess_alphazero.config import Config
    c = Config('mini')
    env = CChessEnv(c)
    env.reset()
    print(env.observation)
    env.step('0001')
    print(env.observation)
    env.step('7770')
    print(env.observation)
    env.render()
    print(env.input_planes()[0+7:3+7])

def test_wxf():
    from cchess_alphazero.environment.light_env.chessboard import L_Chessboard
    lboard = L_Chessboard()
    while not lboard.is_end():
        for i in range(lboard.height):
            print(lboard.screen[i])
        print(f"legal_moves = {lboard.legal_moves()}")
        wxf = input(f'Enter WXF move for {lboard.is_red_turn} r/b: ')
        action = lboard.parse_WXF_move(wxf)
        print(action)
        lboard.move_action_str(action)

def test_sl():
    from cchess_alphazero.worker import sl
    from cchess_alphazero.config import Config
    from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, flip_policy, flip_move
    c = Config('mini')
    labels_n = len(ActionLabelsRed)
    move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
    slworker = sl.SupervisedWorker(c)
    p1 = slworker.build_policy('0001', False)
    print(p1[move_lookup['0001']])
    p2 = slworker.build_policy('0001', True)
    print(p2[move_lookup[flip_move('0001')]])

if __name__ == "__main__":
    test_light()
    
