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
import argparse
import random
import imgaug
from trainer.losses import LossFunc
from dataset import ShapeNet3DData, Bars, ShapeNetDistractor, Pascal1D, ShapeNet1D
from configs.config import Config

"""
Evaluate distractor task with plotting the results
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = Path(__file__)
root_path = file_path.parent
method = "CNPDistractor"

# categories = ['04256520', '04530566']  # ['sofa', 'watercraft']
categories = ['04530566']  # ['sofa', 'watercraft']

labels = {k: int(v) for (v, k) in enumerate(categories)}


def cal_angle_from_sincos(generated_angles):
    angle_sin = generated_angles[..., 0]
    angle_cos = generated_angles[..., 1]
    a_acos = torch.acos(angle_cos)
    angles = torch.where(angle_sin < 0, torch.rad2deg(-a_acos) % 360, torch.rad2deg(a_acos))
    return angles


def calc_error(labels_gt, labels_gen):

    sample_num = labels_gen.size(2)
    labels_gt = labels_gt[:, :, None].repeat(1, 1, sample_num)
    error = torch.stack([torch.abs(labels_gt - labels_gen), torch.abs(labels_gt - (360 + labels_gen)), torch.abs(labels_gen - (360 + labels_gt))], dim=-1)
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


def plot_image_strips(shot_images, generated_mu, ground_truth_images, ground_truth_centers, ground_truth_angles, image_height, image_width, angles_to_plot, output_path):
    shot_num = shot_images.size(0)
    canvas_width = 28 # 15 shot image + 1 space + 12 images spaced every 30 degrees
    canvas_height = 1 # 1 row of generated images + 1 row of ground truth iamges
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
        canvas[0:image_height, column * image_width:(column + 1) * image_width] = ground_truth_images[image_index].squeeze()
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
        plt.text((i+0.5) * image_width, image_height, int(ground_truth_angles[i, 0]), fontsize=7)
    # add ground truth angles
    for i in range(angles_to_plot.shape[0]):
        plt.text((i+16.5) * image_width, image_height, int(angles_to_plot[i, 0]), fontsize=7, color='green')
    # add generated angles
    for i in range(generated_angles.shape[0]):
        plt.text((i+16.5) * image_width, 0, int(generated_angles[i, 0]), fontsize=7, color='blue')
    plt.imshow(canvas, origin='upper', cmap='gray')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_images_and_centers(shot_images, gt_centers, generated_mu, image_height, image_width, output_path):
    shot_num = shot_images.size(0)
    canvas_width = 36 # 15 shot image + 1 space + 12 images spaced every 30 degrees
    canvas_height = 1 # 1 row of generated images + 1 row of ground truth iamges
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

    plt.rcParams["figure.figsize"] = (4608/16, 128/16)
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


def plot_image_and_orientation(shot_images, gt_centers, generated_mu, image_height, image_width, output_path, loss_instances):
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


def evaluate(device, config):
    loss_func = LossFunc(loss_type=config.loss_type, task=config.task)
    # load dataset
    if config.task == 'distractor':
        # used to plot only test category
        data = ShapeNetDistractor(path='./data/distractor',
                                  img_size=config.img_size,
                                  train_fraction=0.8,
                                  val_fraction=0.2,
                                  num_instances_per_item=36,
                                  seed=42,
                                  aug=config.aug_list,
                                  mode='eval',
                                  load_test_categ_only=True,
                                  test_categ=categories)
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
            plot_image_and_center(1.0 - images_to_generate[0], centers_to_generate[0], pr_mu[0], data.get_image_height(), data.get_image_width(), output_path)

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
    i = 15
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    imgaug.seed(config.seed)
    config.max_ctx_num = i
    config.save_path = path + f'/context_num_{i}'
    latent_z_list = evaluate(device, config)

