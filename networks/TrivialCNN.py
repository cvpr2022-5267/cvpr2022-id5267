import torch
from torch import nn
import torchvision.models as models


class TrivialCNN(nn.Module):
    """
    Trivial CNN as baseline
    Image -> Encoder -> r -> Decoder -> bar length
    """
    def __init__(self, kwargs):
        super(TrivialCNN, self).__init__()

        r_dim = kwargs['r_dim']
        # Encoder
        self.f_1 = models.resnet18(pretrained=True, progress=True)
        layers = [
            nn.Linear(1000, r_dim),
            nn.Tanh(),
            nn.Linear(r_dim, r_dim)
        ]
        self.fc_1 = nn.Sequential(*layers)

        # Decoder
        layers_2 = [
            nn.Tanh(),
            nn.Linear(1000 + r_dim, r_dim),
            nn.Tanh(),
            nn.Linear(r_dim, r_dim),
            nn.Tanh(),
            nn.Linear(r_dim, 1)
        ]
        self.xr_to_y = nn.Sequential(*layers_2)

    def forward(self, x):
        """

        :param x: images
        :param target: target label (bar length)
        """
        batch_size, C, H, W = x.size()

        # encode image to r
        x = self.f_1(x)
        r = self.fc_1(x)
        # encode [image, r] to y
        h = torch.cat([x, r], dim=1)
        y_pred = self.xr_to_y(h)
        return y_pred

    @staticmethod
    def loss(y_pred, y_target):
        loss = torch.mean(torch.sqrt((y_pred - y_target) ** 2), dim=0)
        return loss


