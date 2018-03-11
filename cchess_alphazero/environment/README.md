# Chinese Chess Environment

**象棋程序功能 (Functions of the chess program)**

* 判断移动是否合法 (Detect leagal moves)
* 判断胜负 (Detect winner)
* 记录棋谱 (Record)
* 生成训练数据 (Generate training data)
* 可视化 (Visualization)

**API**

Class: `cchess_alphazero.environment.env.ChessEnv`
* `reset()`：重制（初始化）引擎环境

* `observation`：返回当前棋盘

* `copy()`：（深度）复制当前环境

* `input_planes(flip=False)`：返回输入给神经网络的特征平面

  轮到红方时 `flip=True`；轮到黑方时`flip=False`

* `render(gui=False)`：Not Implemented

