import torch
from networks.models import Encoder, NormalEncoder, Decoder
from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


class NeuralProcess(nn.Module):
    """
    Neural Process
    """
    def __init__(self, kwargs):
        super(NeuralProcess, self).__init__()
        r_dim = kwargs['r_dim']
        h_dim = kwargs['h_dim']
        z_dim = kwargs['z_dim']

        self.img_to_r = Encoder(r_dim)
        self.r_to_mu_sigma = NormalEncoder(r_dim, z_dim)
        self.xz_to_y = Decoder(z_dim=z_dim, h_dim=h_dim, y_dim=1)

    @staticmethod
    def aggregate(r_i):
        """
        r_i: latent representation of Encoder for images
        Shape (batch_size, HxW, r_dim)
        """
        # every element in the batch is a sat of images
        # which means batch: (img1,img2..)
        return torch.mean(r_i, dim=0)

    def img_to_mu_sigma(self, img, img_label, aggregate=True):
        """
        Maps (img, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.
        :param img: [batchsize, 3, 256, 256]
        :param img_label: [batchsize, 1]
        :return:
        """
        batch_size, _, _, _ = img.size()
        r = self.img_to_r(img, img_label)
        if aggregate:
            r = self.aggregate(r).unsqueeze(0)
        return self.r_to_mu_sigma(r)

    def forward(self, img_context, y_context, img_target, y_target=None):
        """

        :param img_context: context images
        :param img_target: target image
        :param y_target: target label (bar length)
        :return:
        """
        batch_size, C, H, W = img_context.size()
        _, C_target, H_target, W_target = img_target.size()
        # C_y, H_y, W_y = y_target.size()

        if y_target is not None:
            # y_target is not None means training
            # encode target and context
            # mu_target, sigma_target = self.img_to_mu_sigma(img_target, y_target, aggregate=False)
            mu_target, sigma_target = self.img_to_mu_sigma(img_target, y_target)  # same task should have mean latent
            mu_context, sigma_context = self.img_to_mu_sigma(img_context, y_context)
            # sample from encoded distribution using reparameterization trick
            q_target = Normal(mu_target, sigma_target)
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_target.rsample().repeat(img_target.shape[0], 1)
            # TODO: original paper uses xrz_to_y?
            y_pred_mu, y_pred_sigma = self.xz_to_y(img_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)
            return p_y_pred, q_target, q_context
        else:
            mu_context, sigma_context = self.img_to_mu_sigma(img_context, y_context)
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            z_sample = z_sample.repeat(img_target.shape[0], 1)
            y_pred_mu, y_pred_sigma = self.xz_to_y(img_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)
            return p_y_pred, y_pred_mu, y_pred_sigma

    def loss(self, img_ctx, label_ctx, img_target, label_target):

        p_y_pred, q_target, q_context = self.forward(img_ctx, label_ctx, img_target, label_target)
        log_likelihood = p_y_pred.log_prob(label_target).mean()
        num_tst = img_target.shape[0]
        # q_context = q_context.repeat(num_tst, 1)
        kl = kl_divergence(q_target, q_context).mean()
        return -log_likelihood + kl


