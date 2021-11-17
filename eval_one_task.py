import os.path
import numpy as np
import random
import torch
import imgaug
import argparse

from evaluator.model_evaluator import ModelEvaluator
from trainer.losses import LossFunc
from dataset.refinement import ShapeNet1DRefinement, ShapeNetDistractor
from configs.config import Config


def evaluate(config):
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

    if config.task == 'shapenet_1d':
        data = ShapeNet1DRefinement(path='./data/ShapeNet1D',
                                    img_size=config.img_size,
                                    seed=42,
                                    data_size=config.data_size,
                                    aug=config.aug_list)

    elif config.task == 'distractor':
        data = ShapeNetDistractor(path='./data/distractor',
                                  img_size=config.img_size,
                                  num_instances_per_item=36,
                                  seed=42,
                                  aug=config.aug_list)
    else:
        raise NameError("dataset doesn't exist, check dataset name!")

    loss = LossFunc(loss_type=config.loss_type, task=config.task)

    if 'MAML' not in config.method:
        evaluator = ModelEvaluator(model=model, loss=loss, config=config, data=data)
    else:
        raise NameError(f"method name:{config.method} is not valid!")

    evaluator.evaluate_one_task()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config file")
    args = parser.parse_args()
    config = Config(args.config)
    evaluate(config)


if __name__ == "__main__":
    main()
