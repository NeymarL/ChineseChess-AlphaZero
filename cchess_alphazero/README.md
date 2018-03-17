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
│   │   ├── env.py          : the environment api of the engine
│   │   └── lookup_tables.py 
│   ├── lib                 : helper functions
│   │   ├── data_helper.py  : load & save data
│   │   ├── logger.py       : setup logger
│   │   ├── model_helper.py : load & save model
│   │   └── tf_util.py      : setup tf session
│   ├── play_games          : play with human
│   │   ├── images          : game materials
│   │   ├── play.py         : AI vs human
│   │   ├── test_cli_game.py    : human vs human with cli
│   │   └── test_window_game.py : human vs human with gui
│   ├── worker
│   │   ├── optimize.py     : trainer
│   │   ├── self_play.py    : self play worker
│   │   ├── self_play2.py   : just another implementation of self play worker (can ignore)
│   │   └── evaluator.py    : evaluate with two players
│   ├── config.py           : setup configuration
│   ├── manager.py          : manage to start which worker
│   ├── run.py              : start interface
├── └── test.py             : for debug and test

```
