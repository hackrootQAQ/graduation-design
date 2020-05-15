import json

class CONFIG(object):
    def __init__(self):
        with open("config.json", "r") as f:
            config_dic = json.load(f)
        self.batch_size = config_dic["batch_size"]
        self.embedding_size = config_dic["embedding_size"]
        self.num_comment = config_dic["num_comment"]
        self.num_class = config_dic["num_class"]
        self.lr_base = config_dic["lr_base"]
        self.lr_decay = config_dic["lr_decay"]
        self.lr_step = config_dic["lr_step"]
        self.max_steps = config_dic["max_steps"]
        self.test_interval = config_dic["test_interval"]
        self.attention = config_dic["attention"]

class CONFIG_B(object):
    def __init__(self):
        with open("config_baseline.json", "r") as f:
            config_dic = json.load(f)
        self.batch_size = config_dic["batch_size"]
        self.embedding_size = config_dic["embedding_size"]
        self.num_class = config_dic["num_class"]
        self.gamma = config_dic["gamma"]
        self.lr  = config_dic["lr"]
        self.max_steps = config_dic["max_steps"]
        self.test_interval = config_dic["test_interval"]