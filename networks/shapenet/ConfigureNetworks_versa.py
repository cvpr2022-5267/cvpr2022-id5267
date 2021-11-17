# from resnet import film_resnet18, resnet18
# from adaptation_networks import NullFeatureAdaptationNetwork, FilmAdaptationNetwork, \
#     LinearClassifierAdaptationNetwork, FilmLayerNetwork, FilmArAdaptationNetwork
# from set_encoder import SetEncoder
# from utils import linear_classifier
import torch
import torch.nn.functional as F
import torch.nn as nn
from networks.shapenet.SetEncoder_shapenet import SetEncoder_shapenet, Global_SetEncoder_shapenet
from networks.shapenet.adaptation_networks_shapenet import NullFeatureAdaptationNetwork, FilmAdaptationNetwork, \
    LinearClassifierAdaptationNetwork, FilmLayerNetwork, FilmArAdaptationNetwork
from networks.shapenet.resnet_shapenet import feature_extractor
from networks.shapenet.ConfigureNetworks_shapenet import ViewGenerator


def linear_classifier(x, param_dict):
    """
    Classifier.
    """
    return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])


class ConfigureNetworks_Versa:
    """ Creates the set encoder, feature extractor, feature adaptation, classifier, and classifier adaptation networks.
    """
    def __init__(self, pretrained_resnet_path, batch_normalization, config):
        # self.classifier = linear_classifier

        self.encoder = SetEncoder_shapenet(task_num=config['tasks_per_batch'], label_dim=config['label_dim'], img_channels=config['img_channels'])
        # self.global_encoder = Global_SetEncoder_shapenet(task_num=config['tasks_per_batch'])
        self.view_generator = ViewGenerator(task_num=config['tasks_per_batch'], img_channels=config['img_channels'], img_size=config['img_size'])

        z_g_dim = self.encoder.pre_pooling_fn.output_size

        # parameters for ResNet18
        num_maps_per_layer = [256, 256, 256, 256]
        num_blocks_per_layer = [1, 1, 1, 1]
        num_initial_conv_maps = 256

        self.feature_extractor = feature_extractor(
            pretrained=False,
            pretrained_model_path=pretrained_resnet_path,
            batch_normalization=batch_normalization,
            config=config
        )
        # self.feature_adaptation_network = FilmArAdaptationNetwork(
        #     feature_extractor=self.feature_extractor,
        #     num_maps_per_layer=num_maps_per_layer,
        #     num_blocks_per_layer=num_blocks_per_layer,
        #     num_initial_conv_maps = num_initial_conv_maps,
        #     z_g_dim=z_g_dim
        # )

        # Freeze the parameters of the feature extractor
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False

        # self.classifier_adaptation_network = LinearClassifierAdaptationNetwork(self.feature_extractor.output_size)

    def get_view_generator(self):
        return self.view_generator

    def get_encoder(self):
        return self.encoder

    # def get_global_encoder(self):
    #     return self.global_encoder

    # def get_classifier(self):
    #     return self.classifier

    # def get_classifier_adaptation(self):
    #     return self.classifier_adaptation_network

    # def get_feature_adaptation(self):
    #     return self.feature_adaptation_network

    def get_feature_extractor(self):
        return self.feature_extractor
