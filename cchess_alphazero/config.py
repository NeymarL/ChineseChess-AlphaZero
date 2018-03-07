import os

class Config:

    def __init__(self, config_type="mini"):
        # self.opts = Options()
        # self.resource = ResourceConfig()

        if config_type == "mini":
            import configs.mini as c
        elif config_type == "normal":
            import configs.normal as c
        elif config_type == "distributed":
            import configs.distributed as c
        else:
            raise RuntimeError('unknown config_type: %s' % (config_type))
        self.model = c.ModelConfig()
        self.play = c.PlayConfig()
        self.play_data = c.PlayDataConfig()
        self.trainer = c.TrainerConfig()
        self.eval = c.EvaluateConfig()
        # self.labels = Config.labels
        # self.n_labels = Config.n_labels
        # self.flipped_labels = Config.flipped_labels

