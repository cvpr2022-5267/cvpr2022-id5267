# from resnet import film_resnet18, resnet18
# from adaptation_networks import NullFeatureAdaptationNetwork, FilmAdaptationNetwork, \
#     LinearClassifierAdaptationNetwork, FilmLayerNetwork, FilmArAdaptationNetwork
# from set_encoder import SetEncoder
# from utils import linear_classifier
import torch
import torch.nn.functional as F
import torch.nn as nn
from networks.shapenet.SetEncoder_shapenet import SetEncoder_shapenet, Global_SetEncoder_shapenet, SimplePrePoolNet
from networks.shapenet.adaptation_networks_shapenet import NullFeatureAdaptationNetwork, FilmAdaptationNetwork, \
    LinearClassifierAdaptationNetwork, FilmLayerNetwork, FilmArAdaptationNetwork
from networks.shapenet.resnet_shapenet import film_feature_extractor


def linear_classifier(x, param_dict):
    """
    Classifier.
    """
    return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])


class ViewGenerator(nn.Module):

    def __init__(self, task_num, img_channels, img_size):
        super(ViewGenerator, self).__init__()
        self.task_num = task_num
        self.img_channels = img_channels
        self.img_size = img_size
        self.preprocessing = SimplePrePoolNet(img_channels)
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 1)
        # self.linear2 = nn.Linear(256, 3)
        # self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=1)
        # self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=1)
        # self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=1)
        # self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=(4, 4), stride=(2, 2), padding=1)

    def forward(self, test_images, sample_features):
        num_samples = sample_features.size(1)
        image_per_task = test_images.size(1)
        # preprocessing test images
        test_images = test_images.reshape(self.task_num * image_per_task, self.img_channels, self.img_size[0], self.img_size[1])
        test_images = self.preprocessing(test_images).reshape(self.task_num, image_per_task, -1)
        test_images = test_images[:, :, None, :].repeat(1, 1, num_samples, 1)  # [task, image_per_task, samples, 256]
        sample_features = sample_features[:, None, :, :].repeat(1, image_per_task, 1, 1)  # [task, image_per_task, samples, 256]
        x = torch.cat([test_images, sample_features], dim=-1)
        # x = x.reshape(-1, 259)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.relu(x)

        upper = x.new(torch.ones_like(x) * 360)
        x = torch.where(x > 360, upper, x)
        # x_remain = F.tanh(x[:, :, :, 1:])
        # x = torch.cat([x_degree, x_remain], dim=-1)

        # x = x.reshape(-1, 256, 2, 2)
        # x = F.relu(self.deconv1(x))
        # x = F.relu(self.deconv2(x))
        # x = F.relu(self.deconv3(x))
        # x = F.sigmoid(self.deconv4(x))
        # x = x.reshape(self.task_num, -1, 1, 32, 32)
        return x.squeeze(-1)


class ConfigureNetworks_shapenet:
    """ Creates the set encoder, feature extractor, feature adaptation, classifier, and classifier adaptation networks.
    """
    def __init__(self, pretrained_resnet_path, batch_normalization, config):
        # self.classifier = linear_classifier

        self.encoder = SetEncoder_shapenet(task_num=config['tasks_per_batch'])
        self.global_encoder = Global_SetEncoder_shapenet(task_num=config['tasks_per_batch'])
        self.view_generator = ViewGenerator(task_num=config['tasks_per_batch'])

        z_g_dim = self.encoder.pre_pooling_fn.output_size

        num_maps_per_layer = [256, 256]
        num_blocks_per_layer = [1, 1]
        num_initial_conv_maps = 256

        self.feature_extractor = film_feature_extractor(
            pretrained=False,
            pretrained_model_path=pretrained_resnet_path,
            batch_normalization=batch_normalization
        )
        self.feature_adaptation_network = FilmArAdaptationNetwork(
            feature_extractor=self.feature_extractor,
            num_maps_per_layer=num_maps_per_layer,
            num_blocks_per_layer=num_blocks_per_layer,
            num_initial_conv_maps = num_initial_conv_maps,
            z_g_dim=z_g_dim
        )

        # Freeze the parameters of the feature extractor
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False

        # self.classifier_adaptation_network = LinearClassifierAdaptationNetwork(self.feature_extractor.output_size)

    def get_view_generator(self):
        return self.view_generator

    def get_encoder(self):
        return self.encoder

    def get_global_encoder(self):
        return self.global_encoder

    # def get_classifier(self):
    #     return self.classifier

    # def get_classifier_adaptation(self):
    #     return self.classifier_adaptation_network

    def get_feature_adaptation(self):
        return self.feature_adaptation_network

    def get_feature_extractor(self):
        return self.feature_extractor
