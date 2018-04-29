# Code Structures

```
├── cchess_alphazero
│   ├── agent               : the AI (AlphaZero) agent
│   │   ├── api.py          : neural networks' prediction api
│   │   ├── model.py        : policy & value network model
│   │   └── player.py       : the final agent that play with neural network and MCTS
│   ├── configs             : different types of configuration
│   │   ├── mini.py
│   │   └── normal.py 
│   ├── environment         : a Chinese Chess engine
│   │   │── light_env       : a lightweight chinese chess engine (for training)
│   │   │      ├── chessboard.py
│   │   │      ├── common.py
│   │   │      └── chessman.py.py
│   │   ├── chessboard.py
│   │   ├── chessman.py
│   │   ├── env.py          : the environment api of the engine (mainly used in play-with-human and previous MCTS player)
│   │   ├── static_env.py   : a static chess engine which does not store the board (used in new MCTS player)
│   │   └── lookup_tables.py 
│   ├── lib                 : helper functions
│   │   ├── data_helper.py  : load & save data
│   │   ├── logger.py       : setup logger
│   │   ├── model_helper.py : load & save model
│   │   └── tf_util.py      : setup tf session
│   ├── play_games          : play with human
│   │   ├── images          : game materials
│   │   ├── play.py         : AI vs human with gui
│   │   ├── play_cli.py         : AI vs human with cli
│   │   ├── ob_self_play.py     : observe AI vs AI with cli
│   │   ├── test_cli_game.py    : human vs human with cli
│   │   └── test_window_game.py : human vs human with gui
│   ├── worker
│   │   ├── self_play.py    : self play worker
│   │   ├── self_play_windows.py    : self play worker of Windows os
│   │   ├── compute_elo.py  : evaluate next generation model and compute it's elo
│   │   ├── optimize.py     : trainer
│   │   ├── sl.py           : supervised learning worker
│   │   ├── sl_onegreen.py  : supervised learning worker which train data crawled from game.onegreen.net
│   │   ├── play_with_ucci_engine.py   : play with an ucci engine rather than self play
│   │   └── evaluator.py    : evaluate next generation model with current best model
│   ├── config.py           : setup configuration
│   ├── manager.py          : manage to start which worker
│   ├── run.py              : start interface
│   ├── uci.py              : for UCI protocal
├── └── test.py             : for debug and test

```
