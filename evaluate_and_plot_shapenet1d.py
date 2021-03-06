import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, img_as_ubyte
import argparse
import random
import imgaug
from trainer.losses import LossFunc
from dataset import ShapeNet1D
from configs.config import Config

"""
Evaluate shapenet1d task with plotting the results
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cal_angle_from_sincos(generated_angles):
    angle_sin = generated_angles[..., 0]
    angle_cos = generated_angles[..., 1]
    a_acos = np.arccos(angle_cos)
    angles = np.where(angle_sin < 0, np.rad2deg(-a_acos) % 360, np.rad2deg(a_acos))
    return angles


def plot_image_and_angle(shot_images, gt_y, pr_y, image_height, image_width, output_path):
    shot_num = shot_images.size(0)
    shot_images = shot_images.cpu().numpy()
    pr_y = pr_y.cpu().numpy()
    gt_y = gt_y.cpu().numpy()
    pr_y = cal_angle_from_sincos(pr_y)
    gt_y = cal_angle_from_sincos(gt_y)

    # plot context angles
    for i in range(shot_num):
        plt.rcParams["figure.figsize"] = (8, 8)
        plt.rcParams['savefig.dpi'] = 64
        # plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams.update({'font.size': 35})
        # plt.figure(frameon=False)
        # plt.tight_layout()
        plt.axis('off')
        fig = plt.figure()
        ax = plt.subplot()
        ax.axis('off')

        im = ax.imshow(1.0 - shot_images[i].squeeze(), origin='upper', cmap='gray', vmin=0, vmax=1.0)
        # im = ax.imshow(1.0 - shot_images[i].squeeze(), origin='upper')
        plt.text(0.1 * image_width, image_height - 12, f"gt: {gt_y[i].round(0)}", color='green')
        plt.text(0.1 * image_width, image_height - 4, f"pr: {pr_y[i].round(0)}", color='blue')
        patch = patches.Rectangle((0, 0), 128, 128, transform=ax.transData)
        im.set_clip_path(patch)
        fig.tight_layout()
        plt.savefig(f"{output_path}/{i}", bbox_inches='tight')
        plt.close()


def evaluate(device, config):
    loss_func = LossFunc(loss_type=config.loss_type, task=config.task)
    # load dataset

    if config.task == 'shapenet_1d':
        data = ShapeNet1D(path='./data/ShapeNet1D',
                          img_size=config.img_size,
                          seed=42,
                          data_size=config.data_size,
                          aug=config.aug_list)
    else:
        raise NameError("dataset doesn't exist, check dataset name!")

    import importlib
    module = importlib.import_module(f"networks.{config.method}")
    np_class = getattr(module, config.method)
    model = np_class(config)
    model = model.to(config.device)

    checkpoint = config.checkpoint
    if checkpoint:
        config.logger.info("load weights from " + checkpoint)
        model.load_state_dict(torch.load(checkpoint))
    # model.eval()
    test_iteration = 0

    loss_all = []
    latent_z_list = []
    with torch.no_grad():
        data.gen_bg(config)
        while test_iteration < config.val_iters:
            source = 'test'

            ctx_x, qry_x, ctx_y, qry_y = \
                data.get_batch(source=source, tasks_per_batch=config.tasks_per_batch, shot=config.max_ctx_num)
            ctx_x = ctx_x.to(config.device)
            qry_x = qry_x.to(config.device)
            ctx_y = ctx_y.to(config.device)
            qry_y = qry_y.to(config.device)

            pr_mu, pr_var, sample_z = model(ctx_x, ctx_y, qry_x)
            latent_z_list.append(sample_z)
            loss = loss_func.calc_loss(pr_mu, pr_var, qry_y, test=True)
            loss_all.append(loss.item())

            images_to_generate = qry_x
            centers_to_generate = qry_y
            path_save_image = os.path.join(config.save_path, "image")
            if not os.path.exists(path_save_image):
                os.makedirs(path_save_image)
            output_path = os.path.join(path_save_image, 'output_{0:02d}'.format(test_iteration))
            os.makedirs(output_path)
            plot_image_and_angle(1.0 - images_to_generate[0], centers_to_generate[0], pr_mu[0], data.get_image_height(), data.get_image_width(), output_path)


            test_iteration += 1

    with open(os.path.join(config.save_path, 'losses_all.txt'), 'w') as f:
        np.savetxt(f, loss_all, delimiter=',', fmt='%.4f')
    config.logger.info('Results have been saved to {}'.format(config.save_path))
    config.logger.info('================= Evaluation finished =================\n')
    return latent_z_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config file")
    args = parser.parse_args()
    config = Config(args.config)
    path = config.save_path


    # for i in range(config.max_ctx_num, config.max_ctx_num + 1):
    i = 15

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    imgaug.seed(config.seed)
    config.max_ctx_num = i
    config.save_path = path + f'/context_num_{i}'
    latent_z_list = evaluate(device, config)

