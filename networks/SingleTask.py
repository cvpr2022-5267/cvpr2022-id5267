import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTask(nn.Module):
    """
    Training as single task supervised learning
    """
    def __init__(self, kwargs):
        super(SingleTask, self).__init__()
        self.task_num = kwargs['tasks_per_batch']
        self.layer1 = self._make_conv2d_layer(1, 64)
        self.layer2 = self._make_conv2d_layer(64, 64)
        self.layer3 = self._make_conv2d_layer(64, 64)
        self.layer4 = self._make_conv2d_layer(64, 64)
        self.linear1 = nn.Linear(256, 256)  #TODO: change to xavier initialization
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps):
        return nn.Sequential(
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_maps),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    @staticmethod
    def _make_conv2d_layer_task_norm(in_maps, out_maps):
        return nn.Sequential(
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
            TaskNormI(out_maps),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        x = x.reshape(-1, 1, 32, 32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.relu(x)
        # x_remain = F.tanh(x[:, 1:])
        # x = torch.cat([x_degree, x_remain], dim=-1)
        x = x.reshape(self.task_num, -1)
        return x

    @staticmethod
    def mse_loss(angles_gt, angles_gen):
        """
        Compute the log density under a parameterized normal distribution
        :param inputs: tensor - inputs with axis -1 as random vectors
        :param mu: tensor - mean parameter for normal distribution
        :param logVariance: tensor - log(sigma^2) of distribution
        :return: tensor - log density under a normal distribution
        """
        loss = (angles_gt - angles_gen)**2
        # loss = torch.sum(loss, dim=-1)  # sum loss over dimensions
        loss = torch.mean(loss, dim=-1)  # mean loss over images per task
        loss = torch.mean(loss, dim=-1)  # mean loss over tasks
        return loss

