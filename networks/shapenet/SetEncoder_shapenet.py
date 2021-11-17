import torch
import torch.nn as nn
import torch.nn.functional as F
# from normalization_layers import TaskNormI

"""
    Classes and functions required for Set encoding in adaptation networks. Many of the ideas and classes here are 
    closely related to DeepSets (https://arxiv.org/abs/1703.06114).
"""


def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)


def max_pooling(x):
    return torch.max(x, dim=0, keepdim=True)[0]


def max_pooling_over_tasks(x):
    return torch.max(x, dim=1, keepdim=False)[0]


def mean_pooling_over_tasks(x):
    return torch.mean(x, dim=1, keepdim=False)


class SetEncoder_shapenet(nn.Module):
    """
    Simple set encoder, implementing the DeepSets approach. Used for modeling permutation invariant representations
    on sets (mainly for extracting task-level representations from context sets).
    """
    def __init__(self, task_num, label_dim, img_channels):
        super(SetEncoder_shapenet, self).__init__()
        self.task_num = task_num
        self.label_dim = label_dim
        self.pre_pooling_fn = SimplePrePoolNet(img_channels)
        # self.pooling_fn = mean_pooling
        # self.pooling_fn = max_pooling_over_tasks

    def forward(self, x, label):
        """
        Forward pass through DeepSet SetEncoder. Implements the following computation:

                g(X) = rho ( mean ( phi(x) ) )
                Where X = (x0, ... xN) is a set of elements x in X (in our case, images from a context set)
                and the mean is a pooling operation over elements in the set.

        :param x: (torch.tensor) Set of elements X (e.g., for images has shape batch x C x H x W ).
        :return: (torch.tensor) Representation of the set, single vector in Rk.
        """
        x = self.pre_pooling_fn(x)
        x = torch.cat([x, label], dim=1)
        x = x.reshape(self.task_num, -1, 64 + self.label_dim)
        # x = self.pooling_fn(x)  #TODO: do not use pooling over context??
        return x


class SimplePrePoolNet(nn.Module):
    """
    Simple prepooling network for images. Implements the phi mapping in DeepSets networks. In this work we use a
    multi-layer convolutional network similar to that in https://openreview.net/pdf?id=rJY0-Kcll.
    """
    def __init__(self, img_channels):
        super(SimplePrePoolNet, self).__init__()

        self.layer1 = self._make_conv2d_layer(img_channels, 64)
        self.layer2 = self._make_conv2d_layer(64, 64)
        self.layer3 = self._make_conv2d_layer(64, 64)
        self.layer4 = self._make_conv2d_layer(64, 64)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        # self.linear1 = nn.Linear(64, 64)  # TODO: change to xavier initialization
        # self.linear1 = nn.Linear(64, 256)  #TODO: change to xavier initialization


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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = x.reshape(x.size(0), -1)
        # x = F.relu(self.linear1(x))
        # x = F.elu(self.linear2(x))
        # x = F.elu(self.linear3(x))
        return x

    @property
    def output_size(self):
        return 256



# class SetEncoder_shapenet(nn.Module):
#     """
#     Simple set encoder, implementing the DeepSets approach. Used for modeling permutation invariant representations
#     on sets (mainly for extracting task-level representations from context sets).
#     """
#     def __init__(self, batch_normalization):
#         super(SetEncoder_shapenet, self).__init__()
#         self.pre_pooling_fn = SimplePrePoolNet(batch_normalization)
#         # self.pooling_fn = mean_pooling
#         self.pooling_fn = max_pooling
#
#     def forward(self, x, label):
#         """
#         Forward pass through DeepSet SetEncoder. Implements the following computation:
#
#                 g(X) = rho ( mean ( phi(x) ) )
#                 Where X = (x0, ... xN) is a set of elements x in X (in our case, images from a context set)
#                 and the mean is a pooling operation over elements in the set.
#
#         :param x: (torch.tensor) Set of elements X (e.g., for images has shape batch x C x H x W ).
#         :return: (torch.tensor) Representation of the set, single vector in Rk.
#         """
#         x = self.pre_pooling_fn(x, label)
#         x = self.pooling_fn(x)
#         return x
#

# class SimplePrePoolNet(nn.Module):
#     """
#     Simple prepooling network for images. Implements the phi mapping in DeepSets networks. In this work we use a
#     multi-layer convolutional network similar to that in https://openreview.net/pdf?id=rJY0-Kcll.
#     """
#     def __init__(self, batch_normalization):
#         super(SimplePrePoolNet, self).__init__()
#         if batch_normalization == "task_norm-i":
#             self.layer1 = self._make_conv2d_layer_task_norm(3, 64)
#             self.layer2 = self._make_conv2d_layer_task_norm(64, 64)
#             self.layer3 = self._make_conv2d_layer_task_norm(64, 64)
#             self.layer4 = self._make_conv2d_layer_task_norm(64, 64)
#             self.layer5 = self._make_conv2d_layer_task_norm(64, 64)
#         else:
#             self.layer1 = self._make_conv2d_layer(1, 64)
#             # self.encode_label = nn.Sequential(
#             #     nn.Linear(1, 16),
#             #     nn.ReLU(),
#             #     nn.Linear(16, 16),
#             #     nn.ReLU(),
#             # )
#             self.layer2 = self._make_conv2d_layer(67, 64)
#             self.layer3 = self._make_conv2d_layer(64, 64)
#             self.layer4 = self._make_conv2d_layer(64, 64)
#             # self.layer5 = self._make_conv2d_layer(64, 64)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
#
#     @staticmethod
#     def _make_conv2d_layer(in_maps, out_maps):
#         return nn.Sequential(
#             nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
#             # nn.BatchNorm2d(out_maps),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
#         )
#
#     @staticmethod
#     def _make_conv2d_layer_task_norm(in_maps, out_maps):
#         return nn.Sequential(
#             nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
#             TaskNormI(out_maps),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
#         )
#
#     def forward(self, x, label):
#         x = self.layer1(x)
#         dim_0 = x.shape[-1]
#         # label = self.encode_label(label)  # N x label_dim
#         label = label[:, :, None, None].repeat(1, 1, dim_0, dim_0)
#         x = torch.cat([x, label], dim=1)  # N x (64 + label_dim) x 16 x 16
#
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         # x = self.layer5(x)
#         # x = self.avgpool(x)
#         x1 = self.maxpool(x)
#         x2 = self.avgpool(x)
#         x1 = x1.view(x1.size(0), -1)
#         x2 = x2.view(x2.size(0), -1)
#         x = torch.cat([x1, x2], dim=1)
#         return x
#
#     @property
#     def output_size(self):
#         return 64


class Global_SetEncoder_shapenet(nn.Module):
    """
    Simple set encoder, implementing the DeepSets approach. Used for modeling permutation invariant representations
    on sets (mainly for extracting task-level representations from context sets).
    """
    def __init__(self, task_num, img_channels):
        super(Global_SetEncoder_shapenet, self).__init__()
        self.task_num = task_num
        self.pre_pooling_fn = Global_SimplePrePoolNet(img_channels)
        # self.pooling_fn = mean_pooling
        self.pooling_fn = max_pooling_over_tasks

    def forward(self, x):
        """
        Forward pass through DeepSet SetEncoder. Implements the following computation:

                g(X) = rho ( mean ( phi(x) ) )
                Where X = (x0, ... xN) is a set of elements x in X (in our case, images from a context set)
                and the mean is a pooling operation over elements in the set.

        :param x: (torch.tensor) Set of elements X (e.g., for images has shape batch x C x H x W ).
        :return: (torch.tensor) Representation of the set, single vector in Rk.
        """
        x = self.pre_pooling_fn(x)
        x = x.reshape(self.task_num, -1, 64)
        x = self.pooling_fn(x)
        return x


class Global_SimplePrePoolNet(nn.Module):
    """
    Simple prepooling network for images. Implements the phi mapping in DeepSets networks. In this work we use a
    multi-layer convolutional network similar to that in https://openreview.net/pdf?id=rJY0-Kcll.
    """
    def __init__(self, img_channels):
        super(Global_SimplePrePoolNet, self).__init__()

        self.img_channels = img_channels
        self.layer1 = self._make_conv2d_layer(self.img_channels, 64)
        # self.encode_label = nn.Sequential(
        #     nn.Linear(1, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        #     nn.ReLU(),
        # )
        self.layer2 = self._make_conv2d_layer(64, 64)
        self.layer3 = self._make_conv2d_layer(64, 64)
        self.layer4 = self._make_conv2d_layer(64, 64)
        self.linear1 = nn.Linear(64, 64)  #TODO: change to xavier initialization
        # self.linear2 = nn.Linear(256 + 3, 256)  #TODO: same
        # self.linear3 = nn.Linear(256, 256)
        # self.layer5 = self._make_conv2d_layer(64, 64)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.linear1(x))
        # x = torch.cat([x, label], dim=1)
        # x = F.elu(self.linear2(x))
        # x = F.elu(self.linear3(x))
        return x

    @property
    def output_size(self):
        return 64