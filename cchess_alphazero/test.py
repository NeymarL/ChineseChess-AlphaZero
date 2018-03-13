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
    board = Chessboard()
    board.init_board()
    print(lboard.legal_moves())
    print(lboard.screen)
    # board.move_action_str('0001')
    # lboard.move_action_str('0001')
    # print(board.FENboard())
    # print(lboard.FENboard())
    # print(board.fliped_FENboard())
    # print(lboard.fliped_FENboard())

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

if __name__ == "__main__":
    test_light_env()
    
