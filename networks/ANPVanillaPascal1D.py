import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import save_config
from networks.performer_pytorch import FastAttention

from networks.models import NPDecoder, EncoderFC, AttnLinear


class ANPVanillaPascal1D(nn.Module):
    """
    Conditional Neural Process
    """
    def __init__(self, config):
        super(ANPVanillaPascal1D, self).__init__()
        self.device = config.device
        self.img_size = config.img_size
        self.img_channels = self.img_size[2]
        self.task_num = config.tasks_per_batch
        self.label_dim = config.input_dim
        self.agg_mode = config.agg_mode
        self.img_agg = config.img_agg
        self.y_dim = config.output_dim
        self.dim_w = config.dim_w
        self.n_hidden_units_r = config.n_hidden_units_r
        self.dim_r = config.dim_r
        self.dim_z = config.dim_z
        self.save_latent_z = config.save_latent_z
        seed = config.seed
        torch.manual_seed(seed)  # make network initialization fixed

        # use same architecture as literatures
        self.encoder_w0 = nn.Sequential(
            nn.Conv2d(self.img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4096, self.dim_w)
        )

        self.transform_y = nn.Linear(self.label_dim, self.dim_w // 4)

        self.encoder_r = EncoderFC(input_dim=self.dim_w + self.dim_w // 4,
                                   n_hidden_units_r=self.n_hidden_units_r, dim_r=self.dim_r)

        self.r_to_z = nn.Linear(self.dim_r, self.dim_z)

        self.decoder0 = nn.Sequential(
            nn.Linear(self.dim_w + self.dim_z, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, self.y_dim)
        )

        # self.task_encoder = nn.Sequential(
        #     nn.Linear(256 + self.label_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        # )
        # if self.agg_mode == "baco":
        #     self.rs_to_mu = nn.Linear(256, 256)
        #     self.rs_to_var = nn.Linear(256, 256)

        # self.mu = nn.Linear(256, 256)
        # self.decoder = NPDecoder(aggregate=self.img_agg, output_dim=self.y_dim, task_num=self.task_num, img_channels=self.img_channels, img_size=self.img_size)

        # attention block
        h_dim = self.dim_w
        n_heads = 8
        self._W_k = nn.ModuleList(
            [AttnLinear(h_dim, h_dim) for _ in range(n_heads)]
        )
        self._W_v = nn.ModuleList(
            [AttnLinear(h_dim, h_dim) for _ in range(n_heads)]
        )
        self._W_q = nn.ModuleList(
            [AttnLinear(h_dim, h_dim) for _ in range(n_heads)]
        )
        self._W = AttnLinear(n_heads * h_dim, h_dim)
        self.attn = FastAttention(dim_heads=self.dim_r,
                             # nb_features=nb_features,
                             causal=False)
        self._attention_func = self._multihead_attention
        self.n_heads = n_heads

    def baco(self, mu, r_sigma):
        """

        :param mu: mean value
        :param r_sigma: variance
        :return:
        """
        ctx_num = mu.shape[1]
        self.r_dim = mu.shape[2]
        mu_z = torch.ones(self.task_num, self.r_dim).to(self.device) * 0.0  # initial mu_z is 0, shape [task_num, r_dim]
        sigma_z = torch.ones(self.task_num, self.r_dim).to(self.device) * 1.0  # initial sigma is 1, shape [task_num, r_dim]

        v = mu - mu_z[:, None, :].repeat(1, ctx_num, 1)
        sigma_inv = 1 / r_sigma
        sigma_z = 1 / (1 / sigma_z + torch.sum(sigma_inv, dim=1))
        mu_z = mu_z + sigma_z * torch.sum(sigma_inv * v, dim=1)
        return mu_z, sigma_z

    def _multihead_attention(self, k, v, q):
        k_all = []
        v_all = []
        q_all = []

        for i in range(self.n_heads):
            k_ = self._W_k[i](k)
            v_ = self._W_v[i](v)
            q_ = self._W_q[i](q)

            k_all.append(k_)
            v_all.append(v_)
            q_all.append(q_)

            #out = self._dot_attention(k_, v_, q_)
            #outs.append(out)
        k_all = torch.stack(k_all, dim=1)
        v_all = torch.stack(v_all, dim=1)
        q_all = torch.stack(q_all, dim=1)
        outs=self.attn(q=q_all, k=k_all, v=v_all)
        outs = outs.permute(0,2,3,1).contiguous()
        outs = outs.view(outs.shape[0], outs.shape[1], -1)
        rep = self._W(outs)
        return rep

    def forward(self, batch_train_images, label_train, batch_test_images, test=False):
        """

        :param img_context: context images
        :param img_target: target image
        :param y_target: target label (bar length)
        :return:
        """
        self.test_num = batch_test_images.shape[1]
        self.ctx_num = batch_train_images.shape[1]

        batch_test_images = batch_test_images.reshape(-1, self.img_channels, self.img_size[0], self.img_size[1])
        x_qry = self.encoder_w0(batch_test_images).reshape(self.task_num, self.test_num, self.dim_w)

        if self.ctx_num:
            batch_train_images = batch_train_images.reshape(-1, self.img_channels, self.img_size[0], self.img_size[1])
            x_ctx = self.encoder_w0(batch_train_images).reshape(self.task_num, self.ctx_num, self.dim_w)

            label_train = self.transform_y(label_train)
            x = torch.cat([x_ctx, label_train], dim=2)

            rs = self.encoder_r(x)

            # attention
            if self.agg_mode == 'attention':
                r = self._attention_func(x_ctx, rs, x_qry)
                z = self.r_to_z(r)

            else:
                raise TypeError("agg_mode is not applicable for CNP, choose from ['attention']")
        else:
            z = torch.ones(self.task_num, self.test_num, self.dim_z).to(self.device) * 0.0

        x_qry = torch.cat([x_qry, z], dim=-1)

        pr_y_mu = self.decoder0(x_qry)
        pr_y_var = None

        kl = 0

        if self.save_latent_z:
            return pr_y_mu, pr_y_var, z, kl
        elif not self.save_latent_z:
            return pr_y_mu, pr_y_var, kl

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

    def visualize(self):
        latent_data = self.latent_data.cpu()
        self.TSNE.transform(latent_data[:, :-3], latent_data[:, -3:])
        self.TSNE.vis()

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            if self.init_type == 'normal':
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity=self.activation)
                # if m.bias is not None:
                #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                #     bound = 1 / math.sqrt(fan_in)
                #     init.uniform_(self.bias, -bound, bound)
            elif self.init_type == 'uniform':
                m.weight.data.uniform_(-1.0, 1.0)

                # nn.init.kaiming_uniform_(m.weight, a=0.1, mode='fan_in', nonlinearity=self.activation)

        if isinstance(m, nn.Linear):
            if self.init_type == 'normal':
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity=self.activation)
                # if m.bias is not None:
                #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                #     bound = 1 / math.sqrt(fan_in)
                #     init.uniform_(self.bias, -bound, bound)
            elif self.init_type == 'uniform':
                m.weight.data.uniform_(-1.0, 1.0)

                # nn.init.kaiming_uniform_(m.weight, a=0.1, mode='fan_in', nonlinearity=self.activation)



