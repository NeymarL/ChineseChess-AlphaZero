# ChineseChess-AlphaZero

## About

Chinese Chess reinforcement learning by [AlphaZero](https://arxiv.org/abs/1712.01815) methods.

This project is based on these main resources:
1. DeepMind's Oct 19th publication: [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ).
2. The **great** Reversi/Chess/Chinese chess development of the DeepMind ideas that @mokemokechicken/@Akababa/@TDteach did in their repo: https://github.com/mokemokechicken/reversi-alpha-zero, https://github.com/Akababa/Chess-Zero, https://github.com/TDteach/AlphaZero_ChineseChess
3. A Chinese chess engine with gui: https://github.com/mm12432/MyChess


**Note**: This repo is still under contruction. There is also a slower implementation of MCTS player (previous version), see branch [old](https://github.com/NeymarL/ChineseChess-AlphaZero/tree/old).

## Environment

* Python 3.6.3
* tensorflow-gpu: 1.3.0
* Keras: 2.0.8


## Modules

### Reinforcement Learning

This AlphaZero implementation consists of two workers: `self` and  `opt`.

* `self` is Self-Play to generate training data by self-play using BestModel.
* `opt` is Trainer to train model, and generate new models.

For the sake of faster training (since I don't have 5000 TPUs), another two workers are involved:

* `sl` is Supervised learning to train data crawled from the Internet.
* `eval` is Evaluator to evaluate the NextGenerationModel with the current BestModel.

### GUI

Requirement: PyGame

```bash
python cchess_alphazero/run.py play
```

## How to use

### Setup

### install libraries
```bash
pip install -r requirements.txt
```

If you want to use GPU,

```bash
pip install tensorflow-gpu
```

Make sure Keras is using Tensorflow and you have Python 3.6.3+.

### Configuration

**PlayDataConfig**

* `nb_game_in_file,max_file_num`: The max game number of training data is `nb_game_in_file * max_file_num`.

**PlayConfig, PlayWithHumanConfig**

* `simulation_num_per_move` : MCTS number per move.
* `c_puct`: balance parameter of value network and policy network in MCTS.
* `search_threads`: balance parameter of speed and accuracy in MCTS.
* `dirichlet_alpha`: random parameter in self-play.
* `dirichlet_noise_only_for_legal_moves`: if true, apply dirichlet noise only for legal moves. I don't know whether the DeepMind setting was true or false.
* `vram_frac`: memory use fraction

### Basic Usage

#### Self-Play

```
python cchess_alphazero/run.py self
```

When executed, Self-Play will start using BestModel. If the BestModel does not exist, new random model will be created and become BestModel.

options

* `--new`: create new BestModel
* `--type mini`: use mini config, (see `cchess_alphazero/configs/mini.py`)
* `--gpu '1'`: specify which gpu to use
* `--ucci`: whether to play with ucci engine (rather than self play, see `cchess_alphazero/worker/play_with_ucci_engine.py`)

#### Trainer

```
python cchess_alphazero/run.py opt
```

When executed, Training will start. The current BestModel will be loaded. Trained model will be saved every epoch as new BestModel.

options

* `--type mini`: use mini config, (see `cchess_alphazero/configs/mini.py`)
* `--total-step`: specify total step(mini-batch) numbers. The total step affects learning rate of training.
* `--gpu '1'`: specify which gpu to use

**View training log in Tensorboard**

```
tensorboard --logdir logs/
```

And access `http://<The Machine IP>:6006/`.

#### Play with human

```
python cchess_alphazero/run.py play
```

When executed, the BestModel will be loaded to play against human.

options

* `--ai-move-first`: if set this option, AI will move first, otherwise human move first.
* `--type mini`: use mini config, (see `cchess_alphazero/configs/mini.py`)
* `--gpu '1'`: specify which gpu to use
* `--piece-style`: choose a piece style, default is `WOOD`
* `--bg-style`: choose a board style, default is `CANVAS`
* `--cli`: if set this flag, play with AI in a cli environment rather than gui

#### Evaluator

```
python cchess_alphazero/run.py eval
```

When executed, evaluate the NextGenerationModel with the current BestModel. If the NextGenerationModel does not exist, worker will wait until it exists and check every 5 minutes.

options

* `--type mini`: use mini config, (see `cchess_alphazero/configs/mini.py`)
* `--gpu '1'`: specify which gpu to use

#### Supervised Learning

```
python cchess_alphazero/run.py sl
```

When executed, Training will start. The current SLBestModel will be loaded. Tranined model will be saved every epoch as new SLBestModel.

*About the data*

I have two data sources, one is downloaded from https://wx.jcloud.com/market/packet/10479 ; the other is crawled from http://game.onegreen.net/chess/Index.html (with option --onegreen).

options

* `--type mini`: use mini config, (see `cchess_alphazero/configs/mini.py`)
* `--gpu '1'`: specify which gpu to use
* `--onegreen`: if set the flag, `sl_onegreen` worker will start to train data crawled from `game.onegreen.net`

