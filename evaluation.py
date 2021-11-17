import os.path
import numpy as np
import random
import torch
import imgaug
import argparse

from evaluator.model_evaluator import ModelEvaluator
from trainer.losses import LossFunc
from dataset import ShapeNet3DData, ShapeNetDistractor, Pascal1D, ShapeNet1D
from configs.config import Config


def evaluate(config):
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    imgaug.seed(config.seed)

    import importlib
    module = importlib.import_module(f"networks.{config.method}")
    np_class = getattr(module, config.method)
    model = np_class(config)
    model = model.to(config.device)

    checkpoint = config.checkpoint
    if checkpoint:
        config.logger.info("load weights from " + checkpoint)
        model.load_state_dict(torch.load(checkpoint))

    # load dataset
    if config.task == 'shapenet_3d':
        data = ShapeNet3DData(path='./data/ShapeNet3D_azi180ele30',
                              img_size=config.img_size,
                              train_fraction=0.8,
                              val_fraction=0.2,
                              num_instances_per_item=30,
                              seed=42,
                              aug=config.aug_list,
                              mode='eval')
    elif config.task == 'pascal_1d':
        data = Pascal1D(path='./data/Pascal1D',
                        img_size=config.img_size,
                        seed=42,
                        aug=config.aug_list)
    elif config.task == 'shapenet_1d':
        data = ShapeNet1D(path='./data/ShapeNet1D',
                        img_size=config.img_size,
                        seed=42,
                        data_size=config.data_size,
                        aug=config.aug_list)

    elif config.task == 'distractor':
        data = ShapeNetDistractor(path='./data/distractor',
                                  img_size=config.img_size,
                                  train_fraction=0.8,
                                  val_fraction=0.2,
                                  num_instances_per_item=36,
                                  seed=42,
                                  load_test_categ_only=True,
                                  aug=config.aug_list,
                                  test_categ=['04256520', '04530566'],  # test single category if exceeds cpu memory
                                  mode='eval')
    else:
        raise NameError("dataset doesn't exist, check dataset name!")

    loss = LossFunc(loss_type=config.loss_type, task=config.task)

    if 'MAML' not in config.method:
        evaluator = ModelEvaluator(model=model, loss=loss, config=config, data=data)
    else:
        raise NameError(f"method name:{config.method} is not valid!")

    evaluator.evaluate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config file")
    args = parser.parse_args()
    config = Config(args.config)
    evaluate(config)


if __name__ == "__main__":
    main()
