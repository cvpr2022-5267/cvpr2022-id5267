import torch
from torch import nn
import torch.nn.functional as F
from networks.models import ImageEncoder, NPDecoder


class CNPDistractor(nn.Module):
    """
    Conditional Neural Process
    """
    def __init__(self, config):
        super(CNPDistractor, self).__init__()
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

        self.transform_y = nn.Linear(self.label_dim, self.dim_w)

        self.task_encoder = nn.Sequential(
            nn.Linear(256 + self.dim_w, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        if self.agg_mode == "baco":
            self.latent_mu = nn.Linear(256, 256)
            self.latent_var = nn.Linear(256, 256)

        self.mu = nn.Linear(256, 256)
        self.decoder = NPDecoder(aggregate=self.img_agg, output_dim=self.y_dim, task_num=self.task_num, img_channels=self.img_channels, img_size=self.img_size)

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

    def forward(self, batch_train_images, label_train, batch_test_images, test=False):
        """

        :param img_context: context images
        :param img_target: target image
        :param y_target: target label (bar length)
        :return:
        """

        self.test_num = batch_test_images.shape[1]
        self.ctx_num = batch_train_images.shape[1]
        if self.ctx_num:
            label_train = self.transform_y(label_train)
            batch_train_images = batch_train_images.reshape(-1, self.img_channels, self.img_size[0], self.img_size[1])
            x_ctx = self.img_encoder(batch_train_images)

            x = torch.cat([x_ctx, label_train], dim=2)
            context_features = self.task_encoder(x)
            # aggregate
            if self.agg_mode == 'mean':
                context_features = torch.mean(context_features, dim=1, keepdim=False)
                mu = self.mu(context_features)
                sample_features = mu[:, None, :].repeat(1, self.test_num, 1)
            elif self.agg_mode == 'max':
                context_features = torch.max(context_features, dim=1, keepdim=False)[0]
                mu = self.mu(context_features)
                sample_features = mu[:, None, :].repeat(1, self.test_num, 1)
            elif self.agg_mode == 'baco':
                mu = self.latent_mu(context_features)
                log_variance = self.latent_var(context_features)
                variance = 1e-5 + F.softplus(log_variance)
                mu, log_variance = self.baco(mu, variance)
                mu = self.mu(mu)
                sample_features = mu[:, None, :].repeat(1, self.test_num, 1)
            else:
                raise TypeError("agg_mode is not applicable for CNP, choose from ['mean', 'max', 'baco']")
        else:
            sample_features = torch.ones(self.task_num, self.test_num, 256).to(self.device) * 0.0
            # log_variance = torch.ones(self.task_num, 1, 256).to(self.device) * 1.0

        generated_angles, generated_var = self.decoder(batch_test_images, sample_features)

        kl = 0

        if self.save_latent_z:
            return generated_angles, generated_var, (mu, kl)
        elif not self.save_latent_z:
            return generated_angles, generated_var, kl

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



