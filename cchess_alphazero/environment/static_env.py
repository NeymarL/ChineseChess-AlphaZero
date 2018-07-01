import numpy as np

from cchess_alphazero.environment.light_env.common import *
from cchess_alphazero.environment.lookup_tables import Winner, Fen_2_Idx, flip_move
from logging import getLogger

logger = getLogger(__name__)

INIT_STATE = 'rkemsmekr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RKEMSMEKR'

BOARD_HEIGHT = 10
BOARD_WIDTH = 9

def done(state, turns=-1, need_check=False):
    if 's' not in state:
        return (True, 1, None)
    if 'S' not in state:
        return (True, -1, None)
    # if turns > 0 and turns < 20:
    #     return (False, 0, None)
    board = state_to_board(state)
    red_k, black_k = [0, 0], [0, 0]
    winner = None
    v = 0
    for i in range(BOARD_HEIGHT):
        for j in range(BOARD_WIDTH):
            if board[i][j] == 'k':
                red_k[0] = i
                red_k[1] = j
            if board[i][j] == 'K':
                black_k[0] = i
                black_k[1] = j
    if red_k[0] == 0 and red_k[1] == 0:
        winner = Winner.black
        v = -1
    elif black_k[0] == 0 and black_k[1] == 0:
        winner = Winner.red
        v = 1
    elif red_k[1] == black_k[1]:
        has_block = False
        i = red_k[0] + 1
        while i < black_k[0]:
            if board[i][red_k[1]] != '.':
                has_block = True
                break
            i += 1
        if not has_block:
            v = 1
            winner = Winner.red
    final_move = None
    check = False
    if winner is None:
        legal_moves = get_legal_moves(state, board)
        for mov in legal_moves:
            dest = [int(mov[3]), int(mov[2])]
            if dest == black_k:
                winner = Winner.red
                v = 1
                final_move = mov
                break
    if winner is None and need_check:
        black_state = fliped_state(state)
        black_moves = get_legal_moves(black_state)
        # render(black_state)
        red_k[0] = 9 - red_k[0]
        red_k[1] = 8 - red_k[1]
        for mov in black_moves:
            dest = [int(mov[3]), int(mov[2])]
            if dest == red_k:
                check = True
                # logger.debug(f"Checking move {mov}")
                break
        # logger.debug(f"red_k = {red_k}, black_moves = {black_moves}, check = {check}")
    if need_check:
        return (winner is not None, v, final_move, check)
    else:
        return (winner is not None, v, final_move)

def step(state, action):
    board = state_to_board(state)
    if board[int(action[1])][int(action[0])] == '.':
        raise ValueError(f"No chessman in {action}, state = {state}")
    board[int(action[3])][int(action[2])] = board[int(action[1])][int(action[0])]
    board[int(action[1])][int(action[0])] = '.'
    state = board_to_state(board)
    return fliped_state(state)

def new_step(state, action):
    no_eat = True
    board = state_to_board(state)
    if board[int(action[1])][int(action[0])] == '.':
        raise ValueError(f"No chessman in {action}, state = {state}")
    if board[int(action[3])][int(action[2])] != '.':
        no_eat = False
    board[int(action[3])][int(action[2])] = board[int(action[1])][int(action[0])]
    board[int(action[1])][int(action[0])] = '.'
    state = board_to_state(board)
    return fliped_state(state), no_eat

def evaluate(state):
    piece_vals = {'R': 14, 'K': 7, 'E': 3, 'M': 2, 'S':1, 'C': 5, 'P': 1} # for RED account
    ans = 0.0
    tot = 0
    for c in state:
        if not c.isalpha():
            continue

        if c.isupper():
            ans += piece_vals[c]
            tot += piece_vals[c]
        else:
            ans -= piece_vals[c.upper()]
            tot += piece_vals[c.upper()]
    v = ans / tot
    return np.tanh(v * 3)

def state_to_board(state):
    board = [['.' for col in range(BOARD_WIDTH)] for row in range(BOARD_HEIGHT)]
    x = 0
    y = 9
    for k in range(0, len(state)):
        ch = state[k]
        if ch == ' ':
            break
        if ch == '/':
            x = 0
            y -= 1
        elif ch >= '1' and ch <= '9':
            for i in range(int(ch)):
                board[y][x] = '.'
                x = x + 1
        else:
            board[y][x] = swapcase(ch, s2b=True)
            x = x + 1
    return board

def state_to_planes(state):
    '''
    e.g.
        rkemsmekr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RKEMSMEKR
        rkemsmek1/8r/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RKEMSMEKR
    '''
    planes = np.zeros(shape=(14, 10, 9), dtype=np.float32)
    rows = state.split('/')

    for i in range(len(rows)):
        row = rows[i]
        j = 0
        for letter in row:
            if letter.isalpha():
                # 0 ~ 7 : upper, 7 ~ 14: lower
                planes[Fen_2_Idx[letter] + int(letter.islower()) * 7][i][j] = 1
                j += 1
            else:
                j += int(letter)
    return planes

def state_history_to_planes(state, history):
    '''
    e.g.
        rkemsmekr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RKEMSMEKR
        rkemsmek1/8r/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RKEMSMEKR
    '''
    # logger.debug(f"state = {state}")
    planes = np.zeros(shape=(28, 10, 9), dtype=np.float32)
    rows = state.split('/')
    # 0 ~ 14 for current state
    for i in range(len(rows)):
        row = rows[i]
        j = 0
        for letter in row:
            if letter.isalpha():
                # 0 ~ 7 : upper, 7 ~ 14: lower
                planes[Fen_2_Idx[letter] + int(letter.islower()) * 7][i][j] = 1
                j += 1
            else:
                j += int(letter)
    # 14 ~ 28 for last state
    # history = [...,last state, red action, black state, black action, current state]
    if history and len(history) >= 5:
        last_state = history[-5]
        # logger.debug(f"last state = {last_state}")
        rows = last_state.split('/')
        for i in range(len(rows)):
            row = rows[i]
            j = 0
            for letter in row:
                if letter.isalpha():
                    # 0 ~ 7 : upper, 7 ~ 14: lower
                    planes[Fen_2_Idx[letter] + int(letter.islower()) * 7 + 14][i][j] = 1
                    j += 1
                else:
                    j += int(letter)
    return planes

def board_to_state(board):
    c = 0
    fen = ''
    for i in range(BOARD_HEIGHT - 1, -1, -1):
        c = 0
        for j in range(BOARD_WIDTH):
            if board[i][j] == '.':
                c = c + 1
            else:
                if c > 0:
                    fen = fen + str(c)
                fen = fen + swapcase(board[i][j])
                c = 0
        if c > 0:
            fen = fen + str(c)
        if i > 0:
            fen = fen + '/'
    return fen

def state_to_fen(state, turns):
    fen = ''
    state = "".join([state_to_board_dict[s] if s.isalpha() else s for s in state])
    fen = state + f' w - - 0 {turns}'
    if turns % 2 == 0:  # red turn
        return fen
    else:
        return flip_fen(fen)

def fen_to_state(fen):
    foo = fen.split(' ')
    position = foo[0]
    state = "".join([replace_dict[s] if s.isalpha() else s for s in position])
    return state

def flip_fen(fen):
    foo = fen.split(' ')
    rows = foo[0].split('/')
    def swapcase(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a
    def swapall(aa):
        return "".join([swapcase(a) for a in aa])

    return "/".join([swapall(reversed(row)) for row in reversed(rows)]) \
        + " " + ('w' if foo[1] == 'b' else 'b') \
        + " " + foo[2] \
        + " " + foo[3] + " " + foo[4] + " " + foo[5]

def fliped_state(state):
    rows = state.split('/')
    def swapcase(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a
    def swapall(aa):
        return "".join([swapcase(a) for a in aa])

    return "/".join([swapall(reversed(row)) for row in reversed(rows)])

def get_legal_moves(state, board=None):
    board = board if board is not None else state_to_board(state)
    legal_moves = []
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            ch = board[y][x]
            if not ch.islower():
                continue
            if ch in mov_dir:
                for d in mov_dir[ch]:
                    x_ = x + d[0]
                    y_ = y + d[1]
                    if not can_move(board, x_, y_):
                        continue
                    elif ch == 'p' and y < 5 and x_ != x:  # for red pawn
                        continue
                    elif ch == 'n' or ch == 'b' : # for knight and bishop
                        if board[y+int(d[1]/2)][x+int(d[0]/2)] != '.':
                            continue
                        elif ch == 'b' and y_ > 4:
                            continue
                    elif ch == 'k' or ch == 'a': # for king and advisor
                        if x_ < 3 or x_ > 5:
                            continue
                        if y_ > 2:
                            continue
                    legal_moves.append(move_to_str(x, y, x_, y_))
                    if (ch == 'k'): #for King to King check
                        d, u = y_board_from(board, x, y)
                        if (u < BOARD_HEIGHT and board[u][x] == 'K'):
                            legal_moves.append(move_to_str(x, y, x, u))

            elif ch == 'r' or ch == 'c': # for connon and rook
                l,r = x_board_from(board,x,y)
                d,u = y_board_from(board,x,y)
                for x_ in range(l+1,x):
                    legal_moves.append(move_to_str(x, y, x_, y))
                for x_ in range(x+1,r):
                    legal_moves.append(move_to_str(x, y, x_, y))
                for y_ in range(d+1,y):
                    legal_moves.append(move_to_str(x, y, x, y_))
                for y_ in range(y+1,u):
                    legal_moves.append(move_to_str(x, y, x, y_))
                if ch == 'r': # for rook
                    if can_move(board, l, y):
                        legal_moves.append(move_to_str(x, y, l, y))
                    if can_move(board, r, y):
                        legal_moves.append(move_to_str(x, y, r, y))
                    if can_move(board, x, d):
                        legal_moves.append(move_to_str(x, y, x, d))
                    if can_move(board, x, u):
                        legal_moves.append(move_to_str(x, y, x, u))
                else: # for connon
                    l_, _ = x_board_from(board, l,y)
                    _, r_ = x_board_from(board, r,y)
                    d_, _ = y_board_from(board, x,d)
                    _, u_ = y_board_from(board, x,u)
                    if can_move(board, l_, y):
                        legal_moves.append(move_to_str(x, y, l_, y))
                    if can_move(board, r_, y):
                        legal_moves.append(move_to_str(x, y, r_, y))
                    if can_move(board, x, d_):
                        legal_moves.append(move_to_str(x, y, x, d_))
                    if can_move(board, x, u_):
                        legal_moves.append(move_to_str(x, y, x, u_))
    return legal_moves

def can_move(board, x, y): # basically check the move
    if x < 0 or x > BOARD_WIDTH-1:
        return False
    if y < 0 or y > BOARD_HEIGHT-1:
        return False
    if board[y][x].islower():
        return False
    return True

def x_board_from(board, x, y):
    l = x-1
    r = x+1
    while l > -1 and board[y][l] == '.':
        l = l-1
    while r < BOARD_WIDTH and board[y][r] == '.':
        r = r+1
    return l, r

def y_board_from(board, x, y):
    d = y-1
    u = y+1
    while d > -1 and board[d][x] == '.':
        d = d-1
    while u < BOARD_HEIGHT and board[u][x] == '.':
        u = u+1
    return d, u

def swapcase(a, s2b=False):
    if a.isalpha():
        if s2b:
            a = state_to_board_dict[a]
        else:
            a = replace_dict[a]
        return a.lower() if a.isupper() else a.upper()
    return a

def render(state):
    board = state_to_board(state)
    for i in range(9, -1, -1):
        logger.debug(board[i])
        # print(board[i])

def init(pos):
    board = [['.' for col in range(BOARD_WIDTH)] for row in range(BOARD_HEIGHT)]
    pieces = 'rnbakabnrccpppppRNBAKABNRCCPPPPP'
    position = [pos[i:i+2] for i in range(len(pos)) if i % 2 == 0]
    for pos, piece in zip(position, pieces):
        if pos != '99':
            x, y = int(pos[0]), 9 - int(pos[1])
            board[y][x] = piece
    return board_to_state(board)

def parse_onegreen_move(move):
    x0, y0 = int(move[0]), 9 - int(move[1])
    x1, y1 = int(move[2]), 9 - int(move[3])
    return str(x0) + str(y0) + str(x1) + str(y1)

def parse_ucci_move(move):
    x0, x1 = ord(move[0]) - ord('a'), ord(move[2]) - ord('a')
    move = str(x0) + move[1] + str(x1) + move[3]
    return move

def to_uci_move(action):
    x0, x1 = chr(ord('a') + int(action[0])), chr(ord('a') + int(action[2]))
    move = x0 + action[1] + x1 + action[3]
    return move

def will_check_or_catch(ori_state, action):
    '''
    判断走了下一步是否会造成红方将军或捉子
    '''
    state = step(ori_state, action)     # 判断当前state的红方是否会被将/捉
    board = state_to_board(state)
    # long check
    red_k = [0, 0]
    for i in range(BOARD_HEIGHT):
        for j in range(BOARD_WIDTH):
            if board[i][j] == 'k':
                red_k[0] = i
                red_k[1] = j
    black_state = fliped_state(state)
    black_moves = get_legal_moves(black_state)
    # render(black_state)
    red_k[0] = 9 - red_k[0]
    red_k[1] = 8 - red_k[1]
    for mov in black_moves:
        dest = [int(mov[3]), int(mov[2])]
        if dest == red_k:
            check = True
            # logger.debug(f"Checking move {mov}")
            return True
    # long catch
    first_set = get_catch_list(ori_state)
    second_set = get_catch_list(black_state, black_moves)
    if second_set - first_set != set() and len(second_set) >= len(first_set):
        # logger.debug(f"Long catch, first_set = {first_set}, second_set = {second_set}")
        return True
    else:
        return False

def get_catch_list(state, moves=None):
    catch_list = set()
    if not moves:
        moves = get_legal_moves(state)
    for mov in moves:
        next_state, no_eat = new_step(state, mov)
        if not no_eat:  # 有吃子
            # 判断能不能吃回来(防御)
            could_defend = False
            next_moves = get_legal_moves(next_state)
            fliped_move = flip_move(mov)
            dest = fliped_move[2:]
            for nmov in next_moves:
                if nmov[2:] == dest:
                    could_defend = True
                    break
            if not could_defend:
                i = int(mov[1])
                j = int(mov[0])
                black_board = state_to_board(state)
                if black_board[i][j] == 'p' and i <= 4:
                    continue
                m = int(mov[3])
                n = int(mov[2])
                if black_board[m][n] == 'P' and m > 4:
                    continue
                # print(f"Catch: mov = {mov}, chessman = {black_board[i][j]}")
                catch_list.add((black_board[i][j], i, j, black_board[m][n], m, n))
    return catch_list

def be_catched(state, mov):
    i = int(mov[1])
    j = int(mov[0])
    position = [i, j]
    board = state_to_board(state)
    print(f"Chs = {board[i][j]}")
    black_state = fliped_state(state)
    black_moves = get_legal_moves(black_state)
    position[0] = 9 - position[0]
    position[1] = 8 - position[1]
    for mov in black_moves:
        dest = [int(mov[3]), int(mov[2])]
        if dest == position:
            return True
    return False

def has_attack_chessman(state):
    '''
    INIT_STATE = 'rkemsmekr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RKEMSMEKR'
    '''
    for chessman in state:
        c = chessman.lower()
        if c == 'r' or c == 'k' or c == 'p' or c == 'c':
            return True
    return False
