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

def test_static_env():
    from cchess_alphazero.environment.env import CChessEnv
    import cchess_alphazero.environment.static_env as senv
    from cchess_alphazero.environment.static_env import INIT_STATE
    from cchess_alphazero.environment.lookup_tables import flip_move
    env = CChessEnv()
    env.reset()
    print("env:  " + env.observation)
    print("senv: " + INIT_STATE)
    state = INIT_STATE
    env.step('0001')
    state = senv.step(state, '0001')
    print(senv.evaluate(state))
    print("env:  " + env.observation)
    print("senv: " + state)
    env.step('7770')
    state = senv.step(state, flip_move('7770'))
    print(senv.evaluate(state))
    print("env:  " + env.observation)
    print("senv: " + state)
    env.render()
    board = senv.state_to_board(state)
    for i in range(9, -1, -1):
        print(board[i])
    print("env: ")
    print(env.input_planes()[0+7:3+7])
    print("senv: ")
    print(senv.state_to_planes(state)[0+7:3+7])
    print(f"env:  {env.board.legal_moves()}" )
    print(f"senv: {senv.get_legal_moves(state)}")
    print(set(env.board.legal_moves()) == set(senv.get_legal_moves(state)))

def test_onegreen():
    import cchess_alphazero.environment.static_env as senv
    from cchess_alphazero.environment.lookup_tables import flip_move
    init = '9999299949999999249999869999999958999999519999999999999999997699'
    state = senv.init(init)
    print(state)
    senv.render(state)
    move = senv.parse_onegreen_move('8685')
    state = senv.step(state, move)
    print(state)
    senv.render(state)
    move = senv.parse_onegreen_move('7666')
    state = senv.step(state, flip_move(move))
    print(state)
    senv.render(state)

def test_onegreen2():
    from cchess_alphazero.environment.env import CChessEnv
    import cchess_alphazero.environment.static_env as senv
    from cchess_alphazero.config import Config
    c = Config('mini')
    init = '9999299949999999249999869999999958999999519999999999999999997699'
    env = CChessEnv(c)
    env.reset(init)
    print(env.observation)
    env.render()
    move = senv.parse_onegreen_move('8685')
    env.step(move)
    print(env.observation)
    env.render()
    move = senv.parse_onegreen_move('7666')
    env.step(move)
    print(env.observation)
    env.render()

def test_ucci():
    import cchess_alphazero.environment.static_env as senv
    from cchess_alphazero.environment.lookup_tables import flip_move
    state = senv.INIT_STATE
    state = senv.step(state, '0001')
    fen = senv.state_to_fen(state, 1)
    print(fen)
    senv.render(state)
    move = 'b7b0'
    move = senv.parse_ucci_move(move)
    print(f'Parsed move {move}')
    move = flip_move(move)
    print(f'fliped move {move}')
    state = senv.step(state, move)
    senv.render(state)
    fen = senv.state_to_fen(state, 2)
    print(fen)

def test_done():
    import cchess_alphazero.environment.static_env as senv
    state = '1Rems1e1r/4m4/2c6/p3C1p1p/9/6P2/P3P3P/1c5C1/3p5/2EMSME1R'
    board = senv.state_to_board(state)
    for i in range(9, -1, -1):
        print(board[i])
    print(senv.done(state))

def test_upload():
    from cchess_alphazero.lib.web_helper import upload_file
    from cchess_alphazero.config import Config
    c = Config('mini')
    url = 'http://alphazero.52coding.com.cn/api/upload_game_file'
    path = '/Users/liuhe/Documents/Graduation Project/ChineseChess-AlphaZero/data/play_data/test.json'
    filename = 'test.json'
    data = {'digest': 'test', 'username': 'test'}
    res = upload_file(url, path, filename=filename, data=data)
    print(res)

def test_download():
    from cchess_alphazero.lib.web_helper import download_file
    from cchess_alphazero.config import Config
    c = Config('mini')
    url = 'http://alphazero.52coding.com.cn/model_best_weight.h5'
    path = '/Users/liuhe/Downloads/model_best_weight.h5'
    res = download_file(url, path)
    print(res)

def test_request():
    from cchess_alphazero.lib.web_helper import http_request
    from cchess_alphazero.config import Config
    c = Config('mini')
    url = 'http://alphazero.52coding.com.cn/api/add_model'
    digest = 'd6fce85e040a63966fa7651d4a08a7cdba2ef0e5975fc16a6d178c96345547b3'
    elo = 0
    data = {'digest': digest, 'elo': elo}
    res = http_request(url, post=True, data=data)
    print(res)

def fixbug():
    from cchess_alphazero.config import Config
    from cchess_alphazero.lib.data_helper import get_game_data_filenames, read_game_data_from_file, write_game_data_to_file
    c = Config('distribute')
    files = get_game_data_filenames(c.resource)
    cnt = 0
    for filename in files:
        try:
            data = read_game_data_from_file(filename)
        except:
            print(f"error: {filename}")
            continue
        state = data[0]
        real_data = [state]
        for item in data[1:]:
            action = item[0]
            value = -item[1]
            real_data.append([action, value])
        write_game_data_to_file(filename, real_data)
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)


if __name__ == "__main__":
    fixbug()
    
