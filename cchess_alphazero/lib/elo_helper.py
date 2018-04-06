from logging import getLogger

logger = getLogger(__name__)

# 0 ~ 999: K = 30; 1000 ~ 1999: K = 15; 2000 ~ 2999: K = 10; 3000 ~ : K = 5
K_TABLE = [30, 15, 10, 5]   

R_PRI = 40

def compute_elo(r0, r1, w):
    '''
    Compute the elo rating with method from http://www.xqbase.com/protocol/elostat.htm
    r0: red player's elo rating
    r1: black player's elo rating
    w: game result: 1 = red win, 0.5 = draw, 0 = black win
    '''
    relative_elo = r1 - r0 - R_PRI
    we = 1 / (1 + 10 ** (relative_elo / 400))
    k0 = K_TABLE[-1] if r0 >= 3000 else K_TABLE[r0 // 1000]
    k1 = K_TABLE[-1] if r1 >= 3000 else K_TABLE[r1 // 1000]
    rn0 = int(r0 + k0 * (w - we))
    rn1 = int(r1 + k1 * (we - w))
    rn0 = rn0 if rn0 > 0 else 0
    rn1 = rn1 if rn1 > 0 else 0
    return (rn0, rn1)
