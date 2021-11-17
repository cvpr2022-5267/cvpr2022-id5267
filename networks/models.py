import torch
from torch import nn
from torch.nn import ModuleList
from torch.nn import functional as F
from networks.ResNet import ResNet, BasicBlock
from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)


class EncoderFC(nn.Module):

    def __init__(self, input_dim, n_hidden_units_r, dim_r):
        """
        n_cov_layers <=n where 2**n = image_size
        """
        super(EncoderFC, self).__init__()
        self.input_dim = input_dim
        self.n_hidden_units_r = n_hidden_units_r
        self.dim_r = dim_r

        layers = ModuleList([
            nn.Linear(input_dim, n_hidden_units_r[0]),
            nn.ReLU(inplace=True)
        ])
        for i, unit in enumerate(self.n_hidden_units_r):
            if i == 0:
                pass
            else:
                layers.append(
                    nn.Linear(n_hidden_units_r[i-1], n_hidden_units_r[i]),
                )
                layers.append(
                    nn.ReLU(inplace=True)
                )
        layers.append(
            nn.Linear(n_hidden_units_r[-1], dim_r)
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class ImageEncoder(nn.Module):
    """Maps an (CxHxW) pair to a representation r_i.

    Parameters
    ----------
    C : image channel
    H : image height
    W : image width
    r_dim : int
        Dimension of output representation r.
    """

    def __init__(self, aggregate, task_num, img_channels):
        super(ImageEncoder, self).__init__()
        """
        self.cov_start_dim should smaller than self.cov_end_dim
        self.fc_start_dim should larger than self.fc_end_dim
        """

        self.img_channels = img_channels
        self.task_num = task_num

        # self.normalize = get_normlization(normalize)
        self.aggregate = aggregate
        self.conv1 = nn.Conv2d(self.img_channels, 64, kernel_size=5, stride=2, padding=2,
                               bias=True)

        self.resnet = ResNet(BasicBlock, [1, 1, 1, 1], pretrained=False, progress=True)

    def forward(self, img):
        x = self.conv1(img)
        # x = self.resnet.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        # x = self.resnet.maxpool(x)

        # x = torch.cat([x, label], dim=1)  # N x (32 + label_dim) x 16 x 16

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        if self.aggregate == "mean":
            x = self.resnet.avgpool(x)
        elif self.aggregate == "max":
            x = self.resnet.adaptmax(x)
        elif self.aggregate == "baco":
            x = self.resnet.adaptmax(x)
        elif self.aggregate == "reshape":
            x = x.reshape(x.size(0), -1)
        x = x.reshape(x.size(0), -1)

        x = x.view(self.task_num, -1, x.size(1))

        return x


class NPDecoder(nn.Module):

    def __init__(self, aggregate, output_dim, task_num, img_channels, img_size, pr_unc=False):
        super(NPDecoder, self).__init__()
        """
            pr_unc: predict variance or not 
        """

        self.img_channels = img_channels
        self.task_num = task_num
        self.img_size = img_size
        self.output_dim = output_dim

        # self.normalize = get_normlization(normalize)
        self.aggregate = aggregate
        self.conv1 = nn.Conv2d(self.img_channels, 64, kernel_size=5, stride=2, padding=2, bias=True)

        self.resnet = ResNet(BasicBlock, [1, 1, 1, 1], pretrained=False, progress=True)

        self.fc_mu = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )
        if pr_unc:
            self.fc_var = nn.Sequential(
                nn.Linear(256 + 256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, self.output_dim),
                # nn.Softplus()
            )

    def forward(self, test_images, sample_features, log_variance=None):
        image_per_task = sample_features.size(1)
        test_images = test_images.reshape(self.task_num * image_per_task, self.img_channels, self.img_size[0],
                                          self.img_size[1])
        x = self.conv1(test_images)
        # x = self.resnet.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        # x = self.resnet.maxpool(x)

        # x = torch.cat([x, label], dim=1)  # N x (32 + label_dim) x 16 x 16

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        if self.aggregate == "mean":
            x = self.resnet.avgpool(x)
        elif self.aggregate == "max":
            x = self.resnet.adaptmax(x)
        elif self.aggregate == "baco":
            x = self.resnet.adaptmax(x)
        elif self.aggregate == "reshape":
            x = x.reshape(x.size(0), -1)
        x = x.reshape(x.size(0), -1)

        x = x.reshape(self.task_num, image_per_task, -1)
        x_mu = torch.cat([x, sample_features], dim=-1)
        mu = self.fc_mu(x_mu)
        if log_variance:
            x_var = torch.cat([x, log_variance], dim=-1)
            var = self.fc_var(x_var)
            var = 1e-5 + F.softplus(var)
        else:
            var = None

        return mu, var


class AttnLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        torch.nn.init.normal_(self.linear.weight, std=in_channels ** -0.5)

    def forward(self, x):
        x = self.linear(x)
        return x


def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
            track_running_stats=False)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))


class MetaConvModel(MetaModule):
    """4-layer Convolutional Neural Network architecture from [1].

    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.

    out_features : int
        Number of classes (output of the model).

    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.

    feature_size : int (default: 64)
        Number of features returned by the convolutional head.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True))
        ]))
        self.classifier = MetaLinear(feature_size, out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits


class MetaMLPModel(MetaModule):
    """Multi-layer Perceptron architecture from [1].

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of classes (output of the model).

    hidden_sizes : list of int
        Size of the intermediate representations. The length of this list
        corresponds to the number of hidden layers.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_features, out_features, hidden_sizes):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes
        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i + 1),
            MetaSequential(OrderedDict([
                ('linear', MetaLinear(hidden_size, layer_sizes[i + 1], bias=True)),
                ('relu', nn.ReLU())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = MetaLinear(hidden_sizes[-1], out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits


def ModelConvOmniglot(out_features, hidden_size=64):
    return MetaConvModel(1, out_features, hidden_size=hidden_size,
                         feature_size=hidden_size)


def ModelConvMiniImagenet(out_features, hidden_size=64):
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=5 * 5 * hidden_size)


def ModelMLPSinusoid(hidden_sizes=[40, 40]):
    return MetaMLPModel(1, 1, hidden_sizes)

