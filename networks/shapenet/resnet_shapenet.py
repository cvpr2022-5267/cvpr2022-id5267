"""
This code was based on the file resnet.py (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
from the pytorch/vision library (https://github.com/pytorch/vision).

The original license is included below:

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
# from normalization_layers import TaskNormI

__all__ = ['ResNet', 'resnet18']


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.linear1 = nn.Linear(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.linear1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        return out


class BasicBlockFilm(nn.Module):
    """
    Extension to standard ResNet block (https://arxiv.org/abs/1512.03385) with FiLM layer adaptation. After every batch
    normalization layer, we add a FiLM layer (which applies an affine transformation to each channel in the hidden
    representation). As we are adapting the feature extractor with an external adaptation network, we expect parameters
    to be passed as an argument of the forward pass.
    """
    expansion = 1

    def __init__(self, inplanes, planes):
        super(BasicBlockFilm, self).__init__()
        self.linear1 = nn.Linear(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        # self.linear2 = nn.Linear(planes, planes)

    # def forward(self, x, gamma1, beta1, gamma2, beta2):
    def forward(self, x, gamma1, beta1):
        """
        Implements a forward pass through the FiLM adapted ResNet block. FiLM parameters for adaptation are passed
        through to the method, one gamma / beta set for each convolutional layer in the block (2 for the blocks we are
        working with).
        :param x: (torch.tensor) Batch of images to apply computation to.
        :param gamma1: (torch.tensor) Multiplicative FiLM parameter for first conv layer (one for each channel).
        :param beta1: (torch.tensor) Additive FiLM parameter for first conv layer (one for each channel).
        :param gamma2: (torch.tensor) Multiplicative FiLM parameter for second conv layer (one for each channel).
        :param beta2: (torch.tensor) Additive FiLM parameter for second conv layer (one for each channel).
        :return: (torch.tensor) Resulting representation after passing through layer.
        """
        identity = x

        out = self.linear1(x)
        # out = self.bn1(out)
        out = self._film(out, gamma1, beta1)
        out = self.relu(out)

        # out = self.linear2(out)
        # out = self.bn2(out)
        # out = self._film(out, gamma2, beta2)
        # out = self.relu(out)

        return out

    def _film(self, x, gamma, beta):
        # gamma = gamma[None, :]
        # beta = beta[None, :]
        gamma = gamma[:, None, :].repeat(1, x.size(1), 1)
        beta = beta[:, None, :].repeat(1, x.size(1), 1)
        return gamma * x + beta


class FeatureExtractor(nn.Module):
    def __init__(self, block, layers, bn_fn, config):
        super(FeatureExtractor, self).__init__()
        self.label_dim = config['label_dim']
        self.initial_pool = False
        inplanes = self.inplanes = 64
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=1, bias=False)
        self.linear1 = nn.Linear(64 + self.label_dim, 64)
        # self.bn1 = bn_fn(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])  # only first two layers are different for Versa and CNAPs_shapenet
        self.layer2 = self._make_layer(block, inplanes, layers[1])

        self.layer3 = self._make_layer(BasicBlock, inplanes, layers[2])
        self.layer4 = self._make_layer(BasicBlock, inplanes, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d) or isinstance(m, TaskNormI):
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks):

        layers = []
        layers.append(block(self.inplanes, planes))
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, param_dict=None):
        x = self.linear1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # pooling over instances
        # x = torch.mean(x, dim=1, keepdim=False)  #TODO: add BACO here?
        x = torch.max(x, dim=1, keepdim=False)[0]  #TODO: add BACO here?
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.reshape(x.size(0), -1)

        return x

    def get_layer_output(self, x, param_dict, layer_to_return):
        if layer_to_return == 0:
            x = self.linear1(x)
            # x = self.bn1(x)
            x = self.relu(x)
            if self.initial_pool:
                x = self.maxpool(x)
            return x
        else:
            resnet_layers = [self.layer1, self.layer2]
            # resnet_layers = [self.layer1, self.layer2]
            layer = layer_to_return - 1
            for block in range(self.layers[layer]):
                # x = resnet_layers[layer][block](x, param_dict[layer][block]['gamma1'], param_dict[layer][block]['beta1'],
                #                        param_dict[layer][block]['gamma2'], param_dict[layer][block]['beta2'])
                x = resnet_layers[layer][block](x, param_dict[layer][block]['gamma1'], param_dict[layer][block]['beta1'])
            return x

    @property
    def output_size(self):
        return 256


class FilmFeatureExtractor(FeatureExtractor):
    """
    Wrapper object around BasicBlockFilm that constructs a complete ResNet with FiLM layer adaptation. Inherits from
    ResNet object, and works with identical logic.
    """

    def __init__(self, block, layers, bn_fn):
        FeatureExtractor.__init__(self, block, layers, bn_fn)
        self.layers = layers

    def forward(self, x, param_dict):
        """
        Forward pass through ResNet. Same logic as standard ResNet, but expects a dictionary of FiLM parameters to be
        provided (by adaptation network objects).
        :param x: (torch.tensor) Batch of images to pass through ResNet.
        :param param_dict: (list::dict::torch.tensor) One dictionary for each block in each layer of the ResNet,
                           containing the FiLM adaptation parameters for each conv layer in the model.
        :return: (torch.tensor) Feature representation after passing through adapted network.
        """
        x = self.linear1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        # for block in range(self.layers[0]):
        #     x = self.layer1[block](x, param_dict[0][block]['gamma1'], param_dict[0][block]['beta1'],
        #                            param_dict[0][block]['gamma2'], param_dict[0][block]['beta2'])
        # for block in range(self.layers[1]):
        #     x = self.layer2[block](x, param_dict[1][block]['gamma1'], param_dict[1][block]['beta1'],
        #                            param_dict[1][block]['gamma2'], param_dict[1][block]['beta2'])
        # for block in range(self.layers[2]):
        #     x = self.layer3[block](x, param_dict[2][block]['gamma1'], param_dict[2][block]['beta1'],
        #                            param_dict[2][block]['gamma2'], param_dict[2][block]['beta2'])
        # for block in range(self.layers[3]):
        #     x = self.layer4[block](x, param_dict[3][block]['gamma1'], param_dict[3][block]['beta1'],
        #                            param_dict[3][block]['gamma2'], param_dict[3][block]['beta2'])

        for block in range(self.layers[0]):
            x = self.layer1[block](x, param_dict[0][block]['gamma1'], param_dict[0][block]['beta1'])
        for block in range(self.layers[1]):
            x = self.layer2[block](x, param_dict[1][block]['gamma1'], param_dict[1][block]['beta1'])
        # for block in range(self.layers[2]):
        #     x = self.layer3[block](x, param_dict[2][block]['gamma1'], param_dict[2][block]['beta1'])
        # for block in range(self.layers[3]):
        #     x = self.layer4[block](x, param_dict[3][block]['gamma1'], param_dict[3][block]['beta1'])

        # x = self.avgpool(x)
        x = torch.mean(x, dim=1, keepdim=False)  # add pooling over instances
        x = self.layer3(x)
        x = self.layer4(x)

        # x = x.reshape(x.size(0), -1)

        return x


def get_normalization_layer(batch_normalization):
    if batch_normalization == "task_norm-i":
        nl = TaskNormI
    else:
        nl = nn.BatchNorm2d

    return nl


def feature_extractor(pretrained=False, pretrained_model_path=None, batch_normalization='basic', **kwargs):
    """
        Constructs a ResNet-18 model.
    """
    nl = get_normalization_layer(batch_normalization)

    model = FeatureExtractor(BasicBlock, [1, 1, 1, 1], nl, **kwargs)

    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_dict['state_dict'])
    return model


def film_feature_extractor(pretrained=False, pretrained_model_path=None, batch_normalization="eval", **kwargs):
    """
        Constructs a FiLM adapted ResNet-18 model.
    """
    nl = get_normalization_layer(batch_normalization)

    model = FilmFeatureExtractor(BasicBlockFilm, [1, 1, 1, 1], nl, **kwargs)

    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_dict['state_dict'])

    return model

