import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.ResNet import ResNet, BasicBlock



class SingleTaskResNet(nn.Module):
    """
    Training as single task supervised learning
    """
    def __init__(self, kwargs):
        super(SingleTaskResNet, self).__init__()
        self.task_num = kwargs['tasks_per_batch']
        self.img_channels = kwargs['img_channels']
        self.task_num = kwargs['tasks_per_batch']
        self.img_size = kwargs['img_size']
        self.dataset = kwargs['dataset']

        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.img_channels, 64, kernel_size=5, stride=2, padding=2,
                               bias=True)
        self.resnet = ResNet(BasicBlock, [1, 1, 1, 1], pretrained=False, progress=True)

        self.linear = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.ReLU(),
        )

    def forward(self, img):
        image_per_task = img.size(1)
        x = img.reshape(-1, self.img_channels, self.img_size[0], self.img_size[1])
        x = self.conv1(x)
        # x = self.resnet.bn1(x)
        x = self.activation(x)
        # x = self.resnet.maxpool(x)

        # x = torch.cat([x, label], dim=1)  # N x (32 + label_dim) x 16 x 16

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.adaptmax(x)

        x = x.reshape(x.size(0), -1)
        x = self.linear(x)


        # x_remain = F.tanh(x[:, 1:])
        # x = torch.cat([x_degree, x_remain], dim=-1)
        x = x.reshape(self.task_num, image_per_task, -1)
        return x

    def mse_loss(self, labels_gt, labels_gen):
        """
        Compute the log density under a parameterized normal distribution
        :param inputs: tensor - inputs with axis -1 as random vectors
        :param mu: tensor - mean parameter for normal distribution
        :param logVariance: tensor - log(sigma^2) of distribution
        :return: tensor - log density under a normal distribution
        """
        if self.dataset == 'shapenet':
            # labels_gt = labels_gt[..., 1:]
            labels_gt = labels_gt[..., 0:]
        elif self.dataset == 'bars':
            labels_gt = labels_gt[..., 0:]
        if self.dataset == 'shapenet':
            # labels_gt = labels_gt[:, :, None, None].repeat(1, 1, sample_num, 1)

            loss = torch.sqrt(torch.sum((labels_gt - labels_gen) ** 2, dim=-1))
            # loss = torch.stack([torch.abs(labels_gt - labels_gen), torch.abs(labels_gt - (360 + labels_gen)), torch.abs(labels_gen - (360 + labels_gt))], dim=-1)
            # loss = torch.min(loss, dim=-1)[0]

            # labels_gt = torch.deg2rad(labels_gt[:, :, None])
            # labels_gen = torch.deg2rad(labels_gen)
            # loss = (torch.sin(labels_gt) - torch.sin(labels_gen)) ** 2 + (torch.cos(labels_gt) - torch.cos(labels_gen)) ** 2
            # loss = torch.abs(torch.cos(labels_gt) - torch.cos(labels_gen)) +\
            #        torch.abs(torch.sin(labels_gt) - torch.sin(labels_gen))
        elif self.dataset == 'bars':
            loss = (labels_gt - labels_gen)**2
            loss = torch.mean(loss, dim=-1)  # mean loss over dim
        loss = torch.mean(loss, dim=-1)  # mean loss over images per task
        loss = torch.mean(loss, dim=-1)  # mean loss over tasks
        return loss