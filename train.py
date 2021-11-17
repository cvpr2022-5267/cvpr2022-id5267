import numpy as np
import random
import torch
import imgaug
import argparse

from trainer.model_trainer import ModelTrainer
from trainer.maml_trainer import MAMLTrainer
from trainer.losses import LossFunc
from dataset import ShapeNet3DData, Bars, ShapeNetDistractor, Pascal1D, ShapeNet1D
from configs.config import Config


def train(config):
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

    optimizer_name = config.optimizer
    if config.weight_decay:
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=config.lr, weight_decay=config.beta)
    else:
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=config.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=4e-4)

    # load dataset
    if config.task == 'shapenet_3d':
        data = ShapeNet3DData(path='./data/ShapeNet3D_azi180ele30',
                              img_size=config.img_size,
                              train_fraction=0.8,
                              val_fraction=0.2,
                              num_instances_per_item=30,
                              seed=42,
                              aug=config.aug_list)
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
                                  aug=config.aug_list)
    elif config.task == 'bars':
        data = Bars(path='./data',
                    img_size=config.img_size,
                    train_fraction=0.7,
                    val_fraction=0.1,
                    task_num=config.tasks_per_batch,
                    num_instances_per_item=config.img_num_per_task,
                    round=None,
                    seed=42,
                    mode='train')
    else:
        raise NameError("dataset doesn't exist, check dataset name!")

    loss = LossFunc(loss_type=config.loss_type, task=config.task)

    if 'MAML' not in config.method:
        trainer = ModelTrainer(model=model, loss=loss, optimizer=optimizer, config=config, data=data)
    elif 'MAML' in config.method:
        trainer = MAMLTrainer(model=model,
                              config=config,
                              data=data,
                              optimizer=optimizer,
                              first_order=config.first_order,
                              num_adaptation_steps=config.num_steps,
                              test_num_adaptation_steps=config.test_num_steps,
                              step_size=config.update_lr,  # inner-loop update
                              loss_function=loss,
                              device=config.device
                              )
    else:
        raise NameError(f"method name:{config.method} is not valid!")

    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config file")
    args = parser.parse_args()
    config = Config(args.config)
    train(config)


if __name__ == "__main__":
    main()
