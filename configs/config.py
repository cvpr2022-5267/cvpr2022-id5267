import os
from time import strftime
import logging

import torch
import yaml


class Config(object):

    def __init__(self, config=None):
        if config:
            with open(config, "rb") as f:
                cfg = yaml.safe_load(f)
            self.set_init_values(cfg)

    def set_init_values(self, cfg):
        self.method     = cfg['method']
        self.mode       = cfg['mode'] if 'mode' in cfg.keys() else 'train'
        self.task       = cfg['task']
        self.aug_list   = cfg['aug_list']   # choose from ["MR", "data_aug", "task_aug"]
        self.checkpoint = cfg['checkpoint']
        self.agg_mode   = cfg['agg_mode'] if 'agg_mode' in cfg.keys() else None
        self.img_agg    = cfg['img_agg'] if 'img_agg' in cfg.keys() else None
        self.loss_type  = cfg['loss_type']
        self.save_latent_z   = cfg['save_latent_z'] if 'save_latent_z' in cfg.keys() else False
        self.tasks_per_batch = cfg['tasks_per_batch']
        self.max_ctx_num= cfg['max_ctx_num']

        # params for background image generation online
        self.gen_bg = cfg['gen_bg'] if 'gen_bg' in cfg.keys() else True

        # params for shapenet3d segmentation
        self.output_mask = cfg['output_mask'] if 'output_mask' in cfg.keys() else False

        # params for shapenet1d
        self.data_size      = cfg['data_size'] if 'data_size' in cfg.keys() else None # ["small", "middle" or "large"]

        # params for pascal1d
        self.dim_w      = cfg['dim_w'] if 'dim_w' in cfg.keys() else None
        self.n_hidden_units_r= cfg['n_hidden_units_r'] if 'n_hidden_units_r' in cfg.keys() else None
        self.dim_r      = cfg['dim_r'] if 'dim_r' in cfg.keys() else None
        self.dim_z      = cfg['dim_z'] if 'dim_z' in cfg.keys() else None

        # params for MAML
        self.num_steps  = cfg['num_updates'] if 'num_updates' in cfg.keys() else None
        self.test_num_steps  = cfg['test_num_updates'] if 'test_num_updates' in cfg.keys() else None
        self.dim_hidden = cfg['num_filters'] if 'num_filters' in cfg.keys() else None
        self.first_order= cfg['first_order'] if 'first_order' in cfg.keys() else None
        self.update_lr  = cfg['update_lr'] if 'update_lr' in cfg.keys() else None
        self.beta       = cfg['beta'] if 'beta' in cfg.keys() else 0

        # used for evaluation t-sne
        self.tsne       = cfg['tsne'] if 'tsne' in cfg.keys() else False

        self.noise_scale= cfg['noise_scale']
        self.lr         = cfg['lr']
        self.weight_decay = cfg['weight_decay']
        self.optimizer  = cfg['optimizer']
        self.bg_gen_freq= cfg['bg_gen_freq']
        self.val_iters = cfg['val_iters']
        self.val_freq   = cfg['val_freq']
        self.iterations = cfg['iterations']
        self.device     = torch.device(cfg['device'])
        self.seed       = cfg['seed']
        self.timestamp = strftime("%Y-%m-%d_%H-%M-%S")

        if self.task == 'shapenet_3d' or self.task =='shapenet_3d_segmentation':
            self.img_size = [64, 64, 4]
            self.input_dim = 4
            self.output_dim = 4
        elif self.task == 'pascal_1d':
            self.img_size = [128, 128, 1]
            self.input_dim = 1
            self.output_dim = 1
        elif self.task == 'shapenet_1d':
            self.img_size = [128, 128, 1]
            self.input_dim = 3  # input label [cos(a), sin(a), a]
            self.output_dim = 2  # output [hat_cos(a), hat_sin(a)]
        elif self.task == 'bars':
            self.img_size = [128, 128, 3]
            self.input_dim = 1
            self.output_dim = 1
            # self.img_num_per_task = 30
        elif self.task == 'distractor':
            self.img_size = [128, 128, 1]
            self.input_dim = 2
            self.output_dim = 2
        else:
            raise TypeError(f"{self.task} is not implemented in this experiments!")

        self.save_path = f"results/{self.mode}/{self.method}/{self.timestamp}_{self.task}_datasize_{self.data_size}_{self.agg_mode}_{self.img_agg}{self.loss_type}_{self.aug_list}_seed_{self.seed}"
        self.create_dirs()
        self.save_config()
        self.add_logger()

    def create_dirs(self,):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        os.makedirs(f"{self.save_path}/models", exist_ok=True)

    def save_config(self,):
        with open(os.path.join(self.save_path, "config.yml"), "w") as f:
            yaml.dump(self.__dict__, f)

    def add_logger(self):
        # add logger
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger()
        fh = logging.FileHandler(f'{self.save_path}/log.log', 'a')
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

