class EvaluateConfig:
    def __init__(self):
        self.vram_frac = 1.0
        self.game_num = 4
        self.simulation_num_per_move = 600
        self.thinking_loop = 1
        self.c_puct = 1 # lower  = prefer mean action value
        self.tau_decay_rate = 0
        self.noise_eps = 0.2
        self.max_game_length = 60
        self.max_processes = 5
        self.search_threads = 6
        self.next_generation_replace_rate = 0.55

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
        self.nb_game_in_file = 5
        self.max_file_num = 300
        self.nb_game_save_record = 1


class PlayConfig:
    def __init__(self):
        self.max_processes = 10
        self.search_threads = 8
        self.vram_frac = 1.0
        self.simulation_num_per_move = 800
        self.thinking_loop = 1
        self.logging_thinking = False
        self.c_puct = 1.5
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.2
        self.tau_decay_rate = 0.98
        self.virtual_loss = 3
        self.resign_threshold = -0.95
        self.min_resign_turn = 40
        self.enable_resign_rate = 0.5
        self.max_game_length = 60
        self.share_mtcs_info_in_self_play = False
        self.reset_mtcs_info_per_game = 5


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
        self.momentum = 0.9
        self.loss_weights = [1.0, 1.0]
        self.lr_schedules = [
            (0, 0.01),
            (150000, 0.001),
            (400000, 0.0001),
        ]
        self.sl_game_step = 1000

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
