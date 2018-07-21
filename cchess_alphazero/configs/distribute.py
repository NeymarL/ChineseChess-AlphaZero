class EvaluateConfig:
    def __init__(self):
        self.vram_frac = 1.0
        self.game_num = 10
        self.simulation_num_per_move = 800
        self.thinking_loop = 1
        self.c_puct = 1 # lower  = prefer mean action value
        self.tau_decay_rate = 0.5
        self.noise_eps = 0.1
        self.max_game_length = 200
        self.max_processes = 10
        self.search_threads = 10

    def update_play_config(self, pc):
        pc.simulation_num_per_move = self.simulation_num_per_move
        pc.thinking_loop = self.thinking_loop
        pc.c_puct = self.c_puct
        pc.tau_decay_rate = self.tau_decay_rate
        pc.noise_eps = self.noise_eps
        pc.max_game_length = self.max_game_length
        pc.max_processes = self.max_processes
        pc.search_threads = self.search_threads


class PlayDataConfig:
    def __init__(self):
        self.sl_nb_game_in_file = 250
        self.nb_game_in_file = 1     # WARNING: DO NOT CHANGE THIS PARAMETER
        self.max_file_num = 5000
        self.nb_game_save_record = 1 # not supported in distributed mode


class PlayConfig:
    def __init__(self):
        self.max_processes = 10     # tune this to your cpu cores
        self.search_threads = 10    # increase this will be faster but with weaker performance
        self.vram_frac = 1.0
        self.simulation_num_per_move = 800
        self.thinking_loop = 1
        self.logging_thinking = False
        self.c_puct = 1.5
        self.noise_eps = 0.2
        self.dirichlet_alpha = 0.2
        self.tau_decay_rate = 0.9
        self.virtual_loss = 3
        self.resign_threshold = -0.99
        self.min_resign_turn = 40
        self.enable_resign_rate = 0.99
        self.max_game_length = 200
        self.share_mtcs_info_in_self_play = False
        self.reset_mtcs_info_per_game = 5


class TrainerConfig:
    def __init__(self):
        self.min_games_to_begin_learn = 5000 
        self.min_data_size_to_learn = 0
        self.cleaning_processes = 20
        self.vram_frac = 1.0
        self.batch_size = 1024
        self.epoch_to_checkpoint = 1
        self.dataset_size = 90000000
        self.start_total_steps = 0
        self.save_model_steps = 25
        self.load_data_steps = 100
        self.momentum = 0.9
        self.loss_weights = [1.0, 1.0]
        self.lr_schedules = [
            (0, 0.03),
            (100000, 0.01),
            (200000, 0.003),
            (300000, 0.001),
            (400000, 0.0003),
            (500000, 0.0001),
        ]
        self.sl_game_step = 2000
        self.load_step = 25000

class ModelConfig:
    def __init__(self):
        '''
        WARNING: DO NOT CHANGE THESE PARAMETERS
        '''
        self.cnn_filter_num = 192
        self.cnn_first_filter_size = 5
        self.cnn_filter_size = 3
        self.res_layer_num = 10
        self.l2_reg = 1e-4
        self.value_fc_size = 256
        self.distributed = False
        self.input_depth = 14
