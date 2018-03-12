class EvaluateConfig:
    def __init__(self):
        self.vram_frac = 1.0
        self.game_num = 50
        self.replace_rate = 0.55
        self.play_config = PlayConfig()
        self.play_config.simulation_num_per_move = 600
        self.play_config.thinking_loop = 1
        self.play_config.c_puct = 1 # lower  = prefer mean action value
        self.play_config.tau_decay_rate = 0.6 # I need a better distribution...
        self.play_config.noise_eps = 0
        self.evaluate_latest_first = True
        self.max_game_length = 200 # before: 1000


class PlayDataConfig:
    def __init__(self):
        self.min_elo_policy = 500 # 0 weight
        self.max_elo_policy = 1800 # 1 weight
        self.sl_nb_game_in_file = 250
        self.nb_game_in_file = 30
        self.max_file_num = 200
        self.nb_game_save_record = 100


class PlayConfig:
    def __init__(self):
        self.max_processes = 20
        self.search_threads = 8
        self.vram_frac = 1.0
        self.simulation_num_per_move = 1000
        self.thinking_loop = 1
        self.logging_thinking = False
        self.c_puct = 1.5
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.3
        self.tau_decay_rate = 0.98
        self.virtual_loss = 3
        self.resign_threshold = -1.01
        self.min_resign_turn = 20
        self.max_game_length = 100


class TrainerConfig:
    def __init__(self):
        self.min_games_to_begin_learn = 100
        self.min_data_size_to_learn = 0
        self.cleaning_processes = 4 # RAM explosion...
        self.vram_frac = 1.0
        self.batch_size = 512 # tune this to your gpu memory
        self.epoch_to_checkpoint = 3
        self.dataset_size = 100000
        self.start_total_steps = 0
        self.save_model_steps = 25
        self.load_data_steps = 100
        self.loss_weights = [1.25, 1.0] # [policy, value] prevent value overfit in SL


class ModelConfig:
    def __init__(self):
        self.cnn_filter_num = 256
        self.cnn_first_filter_size = 5
        self.cnn_filter_size = 3
        self.res_layer_num = 7
        self.l2_reg = 1e-4
        self.value_fc_size = 256
        self.distributed = False
        self.input_depth = 14
