import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import save_config
from networks.models import ImageEncoder, NPDecoder
from networks.CBAMEncoder import CBAMEncoder
from utils import LatentVisualizer


class SingleTaskDistractor(nn.Module):
    """
    Conditional Neural Process
    """
    def __init__(self, config):
        super(SingleTaskDistractor, self).__init__()
        self.device = config.device
        self.img_size = config.img_size
        self.img_channels = self.img_size[2] - 1 if config.task == "shapenet_3d" else self.img_size[2]
        self.task_num = config.tasks_per_batch
        self.label_dim = config.input_dim
        self.agg_mode = config.agg_mode
        self.img_agg = config.img_agg
        self.y_dim = config.output_dim
        self.dim_w = config.dim_w
        self.save_latent_z = config.save_latent_z
        seed = config.seed
        torch.manual_seed(seed)  # make network initialization fixed

        self.img_encoder = ImageEncoder(aggregate=self.img_agg, task_num=self.task_num, img_channels=self.img_channels)

        self.task_encoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.mu = nn.Linear(256, 256)
        self.decoder = NPDecoder(aggregate=self.img_agg, output_dim=self.y_dim, task_num=self.task_num, img_channels=self.img_channels, img_size=self.img_size)

    def forward(self, batch_train_images, label_train, batch_test_images, test=False):
        """

        :param img_context: context images
        :param img_target: target image
        :param y_target: target label (bar length)
        :return:
        """

        self.test_num = batch_test_images.shape[1]

        batch_test_images = batch_test_images.reshape(-1, self.img_channels, self.img_size[0], self.img_size[1])
        x = self.img_encoder(batch_test_images)
        x = self.task_encoder(x)
        mu = self.mu(x)
        sample_features = mu

        generated_pos, generated_var = self.decoder(batch_test_images, sample_features)

        kl = 0

        if self.save_latent_z:
            return generated_pos, generated_var, sample_features, kl
        elif not self.save_latent_z:
            return generated_pos, generated_var, kl

    def sample(self, mu, log_variance, test=False):
        if test:
            snum = self.test_num_samples
        else:
            snum = self.num_samples
        return self.sample_normal(mu, log_variance, snum, test)

    def sample_normal(self, mu, log_variance, num_samples, test):
        """
        Generate samples from a parameterized normal distribution.
        :param mu: tf tensor - mean parameter of the distribution.
        :param log_variance: tf tensor - log variance of the distribution.
        :param num_samples: np scalar - number of samples to generate.
        :return: tf tensor - samples from distribution of size num_samples x dim(mu).
        """
        eps = torch.randn(self.task_num, num_samples, mu.size(1)).to(self.device)
        variance = 1e-5 + F.softplus(log_variance)
        variance = variance.repeat(1, num_samples, 1)
        mu = mu.repeat(1, num_samples, 1)
        if test:
            return mu
        else:
            return mu + eps * torch.sqrt(variance)
