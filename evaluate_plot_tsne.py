import json
import os
import math
from pathlib import Path
import torch
import numpy as np
from time import strftime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, img_as_ubyte
from transforms3d.euler import quat2euler
from torch.utils.tensorboard import SummaryWriter
import argparse
import random
import imgaug

from trainer.model_trainer import ModelTrainer
from trainer.maml_trainer import MAMLTrainer
from trainer.losses import LossFunc
from dataset import ShapeNet3DData, Bars, ShapeNetDistractor, Pascal1D, ShapeNet1D
from configs.config import Config
from utils import save_config

"""
Evaluate distractor task with plotting the results, can also save the task latent variable
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = Path(__file__)
root_path = file_path.parent
method = "CNPDistractor"


categories = ['02691156', '02828884', '02933112', '02958343', '02992529', '03001627', '03211117', '03636649', '03691459', '04379243', '04256520', '04530566']
# categories = ['04256520', '04530566']  # ['sofa', 'watercraft']

labels = {k: int(v) for (v, k) in enumerate(categories)}

config = {
    "method": method,
    "dataset": "shapenet_6d",  # "bars" or "shapenet" or "shapenet_6d"
    "img_size": [64, 64],
    "which_aggregate": "attention",
    "loss": "mse_loss",
    "save_latent_z": True,
    "tasks_per_batch": 1,  # number of minibatch
    "num_context_points": 15,  # Number of context points
    "num_samples": 1,
    "test_num_samples": 1,
    "noise_scale": 0.00,
    # coder hyper-parameters
    "encoder_p":
        {
            "label_dim": 16,
            "cov_start_dim": 64,
            "cov_end_dim": 64,
            "n_cov_layers": 4,
            # "fc_start_dim"        : 256,
            # "fc_end_dim"          : 64,
            # "n_fc_layers"         : 3,
            "normalize": "BN",
            "activation": "relu",
            "device": str(device),
        },
    # decoder hyper-parameters
    "decoder_p":
        {
            "pose_target": "quaternion",  # ["quaternion", "translation", "both"]
            "d_cov_start_dim": 64,
            "d_cov_end_dim": 64,
            "d_n_cov_layers": 4,
            # "conv_to_fc": 256,
            # "d_fc_start_dim": 128,
            # "d_fc_end_dim": 64,
            # "d_n_fc_layers": 3,
            "normalize": "BN",
            "activation": "relu",
            "y_dim": 1,
            "device": str(device),
        },
    # "r_dim"               : 128,  # Dimension of output representation r.
    # "h_dim"               : 128,  # Dimension of latent variable z.
    # "z_dim"               : 128,  # Dimension of hidden layer in encoder and decoder. Currently always the same as r_dim
    # "label_dim"           : 64,
    "lr": 1e-4,  # 0.0005931646935963867 for Max
    # "normalize": "BN",
    # "activation": "leaky_relu",
    "optimizer": "Adam",
    # "weight_init_type": "uniform",
    "print_freq": 200,
    "iterations": 5,
    # "test_freq": 1,
    # "dataset": "bar",
    "root_dir": str(root_path) + "/data",
    "checkpoint": str(
        root_path) + f"/results/train/{method}/dataset_shapenet_6d_BACO_determ_cropped_input_quaternion_minimum_loss_test_dim_128/models/best_test_model.pt",
    "device": str(device),
    "seed": 2578,
}


def cal_angle_from_sincos(generated_angles):
    angle_sin = generated_angles[..., 0]
    angle_cos = generated_angles[..., 1]
    a_acos = torch.acos(angle_cos)
    angles = torch.where(angle_sin < 0, torch.rad2deg(-a_acos) % 360, torch.rad2deg(a_acos))
    return angles


def calc_error(labels_gt, labels_gen):
    sample_num = labels_gen.size(2)
    labels_gt = labels_gt[:, :, None].repeat(1, 1, sample_num)
    error = torch.stack([torch.abs(labels_gt - labels_gen), torch.abs(labels_gt - (360 + labels_gen)),
                         torch.abs(labels_gen - (360 + labels_gt))], dim=-1)
    error = torch.min(error, dim=-1)[0]
    error = torch.mean(error, dim=-1)  # mean loss over samples
    error = torch.mean(error, dim=-1)  # mean loss over images per task
    error = torch.mean(error, dim=-1)  # mean loss over tasks
    return error


def calc_distance(centers_to_generate, generated_mu):
    sample_num = generated_mu.size(2)
    centers_to_generate = centers_to_generate[:, :, None, :].repeat(1, 1, sample_num, 1)
    error = torch.sqrt(torch.sum((centers_to_generate - generated_mu) ** 2, dim=-1))
    error = torch.mean(error, dim=1)
    error = torch.mean(error, dim=-1)
    return error


def calc_min_distance(centers_to_generate, generated_mu):
    sample_num = generated_mu.size(2)
    centers_to_generate = centers_to_generate[:, :, None, :].repeat(1, 1, sample_num, 1)
    error = torch.sqrt(torch.sum((centers_to_generate - generated_mu) ** 2, dim=-1))
    e = torch.mean(error, dim=1)  # mean over images not over samples
    _, index = torch.min(e, dim=-1)
    return error, index


def get_angles():
    plot_angles = np.arange(0, 360, 30)  # plot every 30 degrees in azimuth
    angles_to_plot = np.array([plot_angles, np.sin(np.deg2rad(plot_angles)), np.cos(np.deg2rad(plot_angles))]).T
    generate_angles = np.arange(0, 360, 10)  # ask the model to generate views every 10 degrees in azimuth
    angles_to_generate = np.array(
        [generate_angles, np.sin(np.deg2rad(generate_angles)), np.cos(np.deg2rad(generate_angles))]).T
    return angles_to_plot, angles_to_generate


def quaternion_loss(q_gt, q_pr):
    q_pr_norm = np.sqrt(np.sum(q_pr ** 2, axis=-1, keepdims=True))
    q_pr = q_pr / q_pr_norm
    pos_gt_loss = np.abs(q_gt - q_pr).sum(axis=-1)
    neg_gt_loss = np.abs(-q_gt - q_pr).sum(axis=-1)
    L1_loss = np.concatenate([pos_gt_loss[..., None], neg_gt_loss[..., None]], axis=-1)
    L1_loss = np.min(L1_loss, axis=-1)
    L1_loss = L1_loss.mean()
    return L1_loss, q_pr


# def quaternion_loss(q_gt, q_pr):
#     q_pr_norm = np.sqrt(np.sum(q_pr ** 2, axis=-1, keepdims=True))
#     q_pr = q_pr / q_pr_norm
#     index = q_pr[..., 1] < 0
#     q_pr[index, :] *= -1
#     L1_loss = torch.abs(q_gt - q_pr)
#     L1_loss = np.sum(L1_loss, axis=-1).mean()
#     return L1_loss


def plot_image_strips(shot_images, generated_mu, ground_truth_images, ground_truth_centers, ground_truth_angles,
                      image_height, image_width, angles_to_plot, output_path):
    shot_num = shot_images.size(0)
    canvas_width = 28  # 15 shot image + 1 space + 12 images spaced every 30 degrees
    canvas_height = 1  # 1 row of generated images + 1 row of ground truth iamges
    canvas = np.ones((image_height * canvas_height, image_width * canvas_width))
    shot_images = shot_images.cpu().numpy()
    generated_mu = generated_mu.cpu().numpy()

    generated_mu = np.array([generated_mu[np.where(ground_truth_angles[:, 0] == angle)[0]]
                             for angle in angles_to_plot[:, 0]]).squeeze(axis=1)
    # gt_angles = np.array([ground_truth_angles[np.where(ground_truth_angles[:, 0] == angle)[0]]
    #                     for angle in angles_to_plot[:, 0]]).squeeze(axis=1)
    ground_truth_images = np.array([ground_truth_images[np.where(ground_truth_angles[:, 0] == angle)[0]]
                                    for angle in angles_to_plot[:, 0]]).squeeze(axis=1)
    # order images by angle
    generated_mu = generated_mu[np.argsort(angles_to_plot[:, 0], 0)]
    ground_truth_images = ground_truth_images[np.argsort(angles_to_plot[:, 0], 0)]

    blank_image = np.ones(shape=(image_height, image_width))

    # plot the first row which consists of: 1 shot image, 1 blank, 12 generated images equally spaced 30 degrees in azimuth
    # plot the shot image
    for i in range(shot_num):
        canvas[0:image_height, i * image_width:(i + 1) * image_width] = shot_images[i].squeeze()

    # plot 1 blank
    canvas[0:image_height, shot_num * image_width:16 * image_width] = blank_image.repeat((16 - shot_num), axis=1)

    # plot generated images
    image_index = 0
    for column in range(16, canvas_width):
        canvas[0:image_height, column * image_width:(column + 1) * image_width] = ground_truth_images[
            image_index].squeeze()
        image_index += 1

    # plot the ground truth strip in the 2nd row
    # Plot 2 blanks
    # k = 0
    # for column in range(0, 2):
    #     canvas[image_height:2 * image_height, column * image_width:(column + 1) * image_width] = blank_image
    # Plot ground truth images
    # image_index = 0
    # for column in range(2, canvas_width):
    #     canvas[image_height:2 * image_height, column * image_width:(column + 1) * image_width] = ground_truth_images[image_index].squeeze()
    #     image_index += 1

    plt.figure(figsize=(8, 10), frameon=False)
    plt.axis('off')
    # plot context angles
    for i in range(shot_num):
        plt.text((i + 0.5) * image_width, image_height, int(ground_truth_angles[i, 0]), fontsize=7)
    # add ground truth angles
    for i in range(angles_to_plot.shape[0]):
        plt.text((i + 16.5) * image_width, image_height, int(angles_to_plot[i, 0]), fontsize=7, color='green')
    # add generated angles
    for i in range(generated_angles.shape[0]):
        plt.text((i + 16.5) * image_width, 0, int(generated_angles[i, 0]), fontsize=7, color='blue')
    plt.imshow(canvas, origin='upper', cmap='gray')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_images_and_centers(shot_images, gt_centers, generated_mu, image_height, image_width, output_path):
    shot_num = shot_images.size(0)
    canvas_width = 36  # 15 shot image + 1 space + 12 images spaced every 30 degrees
    canvas_height = 1  # 1 row of generated images + 1 row of ground truth iamges
    canvas = np.ones((image_height * canvas_height, image_width * canvas_width))
    shot_images = shot_images.cpu().numpy()
    generated_mu = generated_mu.cpu().numpy()
    gt_centers = gt_centers.cpu().numpy()

    # plot the first row which consists of: 1 shot image, 1 blank, 12 generated images equally spaced 30 degrees in azimuth
    # plot the shot image
    for i in range(shot_num):
        # img = shot_images[i].squeeze()
        # img[gen_center[1] - 3: gen_center[1] + 3, gen_center[0] - 3: gen_center[0] + 3] =
        canvas[0:image_height, i * image_width:(i + 1) * image_width] = shot_images[i].squeeze()

    plt.rcParams["figure.figsize"] = (4608 / 16, 128 / 16)
    plt.rcParams['savefig.dpi'] = 16
    plt.figure(frameon=False)
    plt.tight_layout()
    plt.axis('off')

    # plot context angles
    for i in range(shot_num):
        gen_center = (generated_mu[i, :, 0] + i * image_width, generated_mu[i, :, 1])
        gt_center = (gt_centers[i][0] + i * image_width, gt_centers[i][1])
        plt.plot(*gen_center, 'bo', markersize=7)
        plt.plot(*gt_center, marker='o', markersize=7, color='green')
        # plt.text((i+0.5) * image_width, image_height, int(ground_truth_angles[i, 0]), fontsize=7)

    plt.imshow(canvas, origin='upper', cmap='gray')
    plt.savefig(output_path)
    plt.close()


def plot_image_and_center(shot_images, gt_centers, generated_mu, image_height, image_width, output_path):
    shot_num = shot_images.size(0)
    shot_images = shot_images.cpu().numpy()
    generated_mu = generated_mu.cpu().numpy()
    gt_centers = gt_centers.cpu().numpy()

    # plt.rcParams["figure.figsize"] = (4608/16, 128/16)
    # plt.rcParams['savefig.dpi'] = 16
    # plt.figure(frameon=False)

    # plot context angles
    for i in range(shot_num):
        gen_center = (generated_mu[i, 0], generated_mu[i, 1])
        gt_center = (gt_centers[i, 0], gt_centers[i, 1])

        fig = plt.figure(figsize=(2, 2))
        ax = plt.subplot()
        ax.axis('off')

        im = ax.imshow(shot_images[i].squeeze(), origin='upper', cmap='gray')
        ax.plot(*gen_center, 'bo', markersize=7)
        ax.plot(*gt_center, marker='o', markersize=7, color='green')
        patch = patches.Rectangle((0, 0), 128, 128, transform=ax.transData)
        im.set_clip_path(patch)
        fig.tight_layout()
        plt.savefig(f"{output_path}/{i}")
        plt.close()


def plot_image_and_orientation(shot_images, gt_centers, generated_mu, image_height, image_width, output_path,
                               loss_instances):
    shot_num = shot_images.size(0)
    # canvas_width = 30 # 15 shot image + 1 space + 12 images spaced every 30 degrees
    # canvas_height = 1 # 1 row of generated images + 1 row of ground truth iamges
    # canvas = np.ones(((image_height + 20) * canvas_height, image_width * canvas_width, 3))
    shot_images = shot_images.cpu().numpy()
    generated_mu = generated_mu.cpu().numpy().squeeze()
    gt_centers = gt_centers[:, :4].cpu().numpy()

    # plot the first row which consists of: 1 shot image, 1 blank, 12 generated images equally spaced 30 degrees in azimuth
    # plot the shot image
    for i in range(shot_num):
        plt.rcParams["figure.figsize"] = (8, 8)
        plt.rcParams['savefig.dpi'] = 64
        plt.rcParams['axes.facecolor'] = 'white'
        # plt.figure(frameon=False)
        # plt.tight_layout()
        plt.axis('off')

        canvas = shot_images[i].squeeze().transpose(1, 2, 0)
        gen_q = generated_mu[i, :]
        gt_q = gt_centers[i, :]
        q_loss, gen_q = quaternion_loss(gt_q, gen_q)
        gen_euler = np.rad2deg(quat2euler(gen_q)) % 360
        gt_euler = np.rad2deg(quat2euler(gt_q)) % 360
        plt.text(0.1 * image_width, image_height + 2, f"gt: {gt_euler.round(2)}", color='green')
        plt.text(0.1 * image_width, image_height + 4, f"pr: {gen_euler.round(2)}", color='blue')
        plt.text(0.1 * image_width, image_height + 6, f"loss: {q_loss:.3f}", color='blue')
        plt.text(0.1 * image_width, image_height + 8, f"gt_q: {gt_q.round(2)}", color='green')
        plt.text(0.5 * image_width, image_height + 8, f"pr_q: {gen_q.round(2)}", color='blue')

        loss_instances.append(q_loss)
        plt.imshow(canvas, origin='upper')
        plt.savefig(f"{output_path}/{i}")
        plt.close()

    with open(os.path.join(output_path, 'loss_instances.txt'), 'w') as f:
        np.savetxt(f, np.array(loss_instances).round(4), delimiter=',', fmt='%.4f')


def save_images_to_folder(generated_images, generated_angles, ground_truth_images, ground_truth_angles, path):
    """
    save_images_to_folder: Saves view reconstruction images to a folder.
    """
    generated_images = generated_images.cpu().numpy()
    # order the images according to ascending angle
    ground_truth_images = ground_truth_images[np.argsort(ground_truth_angles[:, 0], 0)]
    generated_images = generated_images[np.argsort(generated_angles[:, 0], 0)]

    if not os.path.exists(path):
        os.makedirs(path)

    counter = 0
    for (im_gt, im_gen) in zip(ground_truth_images, generated_images):
        ground_truth_path = os.path.join(path, 'ground_truth_{0:02d}.png'.format(counter))
        io.imsave(ground_truth_path, img_as_ubyte(im_gt.squeeze()))
        generated_path = os.path.join(path, 'generated_{0:02d}.png'.format(counter))
        io.imsave(generated_path, img_as_ubyte(im_gen.squeeze()))
        counter += 1


def evaluate(device, config, categ=None):
    loss_func = LossFunc(loss_type=config.loss_type, task=config.task)
    # load dataset
    if config.task == 'shapenet_3d':
        data = ShapeNet3DData(path='./data/ShapeNet3DSimpler',
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

        # used to visualize task latent variable for further T-SNE
        data = ShapeNetDistractor(path='./data/distractor',
                                  img_size=config.img_size,
                                  train_fraction=0.8,
                                  val_fraction=0.2,
                                  num_instances_per_item=36,
                                  seed=42,
                                  aug=config.aug_list,
                                  mode='eval',
                                  load_test_categ_only=True,
                                  test_categ=categ)

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

    angles_to_plot, angles_generate = get_angles()
    loss_all = []
    latent_z_list = []
    with torch.no_grad():
        data.gen_bg(config)
        while test_iteration < config.val_iters:

            if config.tsne:
                source = random.choice(['test', 'validation'])
            else:
                source = 'test'

            ctx_x, qry_x, ctx_y, qry_y = \
                data.get_batch(source=source, tasks_per_batch=config.tasks_per_batch, shot=config.max_ctx_num)
            ctx_x = ctx_x.to(config.device)
            qry_x = qry_x.to(config.device)
            ctx_y = ctx_y.to(config.device)
            qry_y = qry_y.to(config.device)

            pr_mu, pr_var, (sample_z, _) = model(ctx_x, ctx_y, qry_x)
            latent_z_list.append(sample_z)
            loss = loss_func.calc_loss(pr_mu, pr_var, qry_y, test=True)
            print(f"Iteration: {test_iteration}, losss; {loss}")
            loss_all.append(loss.item())

            if config.task == 'shapenet_6d':
                batch_train_images = convert_channel_last_np_to_tensor(batch_train_images[..., :3]).to(
                    device)  # using only rbg channels
                batch_test_images = convert_channel_last_np_to_tensor(batch_test_images[..., :3]).to(device)
                batch_train_Q = torch.from_numpy(batch_train_Q).type(torch.FloatTensor).to(device)
                batch_test_Q = torch.from_numpy(batch_test_Q).type(torch.FloatTensor).to(device)
                batch_train_T = torch.from_numpy(batch_train_T).type(torch.FloatTensor).to(device)
                batch_test_T = torch.from_numpy(batch_test_T).type(torch.FloatTensor).to(device)
                label_train = torch.cat((batch_train_Q, batch_train_T), dim=-1)
                label_test = torch.cat((batch_test_Q, batch_test_T), dim=-1)

            # images_to_generate = qry_x
            # centers_to_generate = qry_y
            # path_save_image = os.path.join(config.save_path, "image")
            # if not os.path.exists(path_save_image):
            #     os.makedirs(path_save_image)
            # output_path = os.path.join(path_save_image, 'output_{0:02d}'.format(test_iteration))
            # os.makedirs(output_path)
            # plot_image_and_center(1.0 - images_to_generate[0], centers_to_generate[0], pr_mu[0],
            #                       data.get_image_height(), data.get_image_width(), output_path)
            # # plot_image_and_orientation(images_to_generate[0], centers_to_generate[0], pr_mu[0],
            # #                        data.get_image_height(), data.get_image_width(), output_path, loss_instances)

            test_iteration += 1

    # with open(os.path.join(config.save_path, 'losses_all.txt'), 'w') as f:
    #     np.savetxt(f, loss_all, delimiter=',', fmt='%.4f')

    return latent_z_list, np.array(loss_all).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config file")
    args = parser.parse_args()
    config = Config(args.config)
    path = config.save_path
    latent_all_categ = np.zeros((0, 257))
    losses_all_categ = []

    for categ in categories:
        for i in range(config.max_ctx_num, config.max_ctx_num + 1):
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(config.seed)
            random.seed(config.seed)
            np.random.seed(config.seed)
            imgaug.seed(config.seed)

            config.max_ctx_num = i
            config.save_path = path + f'/context_num_{i}'
            if not os.path.exists(config.save_path):
                os.makedirs(config.save_path)

            # latent_np = np.array([]).reshape(0, 65)
            # for categ in categories:

            latent_z_list, loss = evaluate(device, config, categ=[categ])
            losses_all_categ.append(loss)
            latent_z = torch.stack(latent_z_list).cpu().numpy().squeeze()
            l = np.ones((latent_z.shape[0], 1)) * labels[categ]
            latent_z = np.concatenate((latent_z, l), axis=1)
            latent_all_categ = np.vstack((latent_all_categ, latent_z))

    with open(f'{config.save_path}/latent_all_categ.npy', 'wb') as f:
        np.save(f, latent_all_categ)

    with open(f'{config.save_path}/losses_all_categ.txt', 'wb') as f:
        np.savetxt(f, np.array(losses_all_categ))

    with open(f'{config.save_path}/losses_all_categ_mean.txt', 'wb') as f:
        np.savetxt(f, np.array(losses_all_categ).mean(keepdims=True))
