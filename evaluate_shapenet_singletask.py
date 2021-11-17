import json
import os
import math
from pathlib import Path
import torch
import numpy as np
from time import strftime
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from torch.utils.tensorboard import SummaryWriter

from networks.NeuralProcess import NeuralProcess
from dataset.dataset import BarImage, bar
from train import convert_images_label, split_data
from networks import NeuralProcess, ANP, CondNeuralProcess, TrivialCNN, BNP, MetaFun, CNAPs_shapenet, Versa
from utils import save_config
from dataset.shapenet_distractor import ShapeNetData
from dataset.bars import Bars
from train_shapenet import convert_channel_last_np_to_tensor, augment_images_shapenet, augment_images_bars

"""
Validate the trained model with different color bar that never seen during training
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = Path(__file__)
root_path = file_path.parent
method = "SingleTaskResNet"

kwargs = {
    "method": method,
    "dataset": 'shapenet',
    "img_size": [128, 128],
    "which_aggregate": "max",
    "tasks_per_batch": 1,  # number of minibatch
    "num_context_points": 2,  # Number of context points
    "num_samples": 1,
    "noise_scale": 0.00,
    # coder hyper-parameters
    "encoder_p":
        {
            "label_dim": 16,
            "cov_start_dim": 64,
            "cov_end_dim": 64,
            "n_cov_layers": 4,
            "normalize": "BN",
            "activation": "relu",
            "device": str(device),
        },
    # decoder hyper-parameters
    "decoder_p":
        {
            "d_cov_start_dim": 64,
            "d_cov_end_dim": 64,
            "d_n_cov_layers": 4,
            "normalize": "BN",
            "activation": "relu",
            "y_dim": 1,
            "device": str(device),
        },
    "lr": 1e-4,  # 0.0005931646935963867 for Max
    "optimizer": "Adam",
    "print_freq": 200,
    "iterations": 199,
    "root_dir": str(root_path) + "/data",
    "checkpoint": str(root_path) + f"/results/train/{method}/non_truncated_distractor/models/best_val_model.pt",
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
    error = torch.stack([torch.abs(labels_gt - labels_gen), torch.abs(labels_gt - (360 + labels_gen)), torch.abs(labels_gen - (360 + labels_gt))], dim=-1)
    error = torch.min(error, dim=-1)[0]
    error = torch.mean(error, dim=-1)  # mean loss over samples
    error = torch.mean(error, dim=-1)  # mean loss over images per task
    error = torch.mean(error, dim=-1)  # mean loss over tasks
    return error


def calc_distance(centers_to_generate, generated_centers):
    error = torch.sqrt(torch.sum((centers_to_generate - generated_centers) ** 2, dim=-1))
    error = torch.mean(error, dim=-1)
    error = torch.mean(error, dim=-1)
    return error


def get_angles():
    plot_angles = np.arange(0, 360, 30)  # plot every 30 degrees in azimuth
    angles_to_plot = np.array([plot_angles, np.sin(np.deg2rad(plot_angles)), np.cos(np.deg2rad(plot_angles))]).T
    generate_angles = np.arange(0, 360, 10)  # ask the model to generate views every 10 degrees in azimuth
    angles_to_generate = np.array(
        [generate_angles, np.sin(np.deg2rad(generate_angles)), np.cos(np.deg2rad(generate_angles))]).T
    return angles_to_plot, angles_to_generate


def plot_image_and_centers(shot_images, gt_centers, generated_centers, image_height, image_width, output_path):
    shot_num = shot_images.size(0)
    canvas_width = 36 # 15 shot image + 1 space + 12 images spaced every 30 degrees
    canvas_height = 1 # 1 row of generated images + 1 row of ground truth iamges
    canvas = np.ones((image_height * canvas_height, image_width * canvas_width))
    shot_images = shot_images.cpu().numpy()
    generated_centers = generated_centers.cpu().numpy().squeeze()
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
        gen_center = (generated_centers[i][0] + i * image_width, generated_centers[i][1])
        gt_center = (gt_centers[i][0] + i * image_width, gt_centers[i][1])
        plt.plot(*gen_center, marker='o', markersize=7, color='blue')
        plt.plot(*gt_center, marker='o', markersize=7, color='green')
        # plt.text((i+0.5) * image_width, image_height, int(ground_truth_angles[i, 0]), fontsize=7)

    plt.imshow(canvas, origin='upper', cmap='gray')
    plt.savefig(output_path)
    plt.close()


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


def evaluate(device, kwargs):
    # load dataset
    if kwargs['dataset'] == 'shapenet':
        data = ShapeNetData(path='./data',
                            img_size=kwargs['img_size'],
                            train_fraction=0.7,
                            val_fraction=0.1,
                            num_instances_per_item=36,
                            seed=42,
                            mode='test')
        kwargs['label_dim'] = 2
        kwargs['decoder_p']['y_dim'] = 2
    elif kwargs['dataset'] == 'bars':
        data = Bars(path='./data',
                    img_size=kwargs['img_size'],
                    train_fraction=0.7,
                    val_fraction=0.1,
                    num_instances_per_item=30,
                    seed=42,
                    mode='test')
        kwargs['label_dim'] = 1
        kwargs['decoder_p']['y_dim'] = 1
    kwargs['img_channels'] = data.get_image_channels()

    import importlib
    module = importlib.import_module(f"networks.{kwargs['method']}")
    np_class = getattr(module, kwargs['method'])
    model = np_class(kwargs).to(device)
    checkpoint = kwargs["checkpoint"]
    if checkpoint:
        print("load weights from " + checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location='cuda:0'))
    else:
        raise ValueError("No model loaded, Load weights from training directory!")
        exit()
    # model.eval()
    test_iteration = 0
    epoch_loss = 0

    angles_to_plot, angles_generate = get_angles()
    # angles_to_generate = torch.Tensor(angles_generate).to(device)[None, :, :].repeat(kwargs['tasks_per_batch'], 1, 1)
    loss_all = []
    error_all = []
    with torch.no_grad():
        while test_iteration < kwargs['iterations']:
            batch_train_images, batch_test_images, batch_train_angles, batch_test_angles, batch_train_centers, batch_test_centers = \
                data.get_batch(source='test', tasks_per_batch=kwargs['tasks_per_batch'],
                               shot=kwargs['num_context_points'])

            if kwargs['dataset'] == 'shapenet':
                batch_train_images = convert_channel_last_np_to_tensor(batch_train_images).to(device)
                batch_test_images = convert_channel_last_np_to_tensor(batch_test_images).to(device)
                batch_train_angles = torch.from_numpy(batch_train_angles).type(torch.FloatTensor).to(device)
                batch_test_angles = torch.from_numpy(batch_test_angles).type(torch.FloatTensor).to(device)
                batch_train_centers = torch.from_numpy(batch_train_centers).type(torch.FloatTensor).to(device)
                batch_test_centers = torch.from_numpy(batch_test_centers).type(torch.FloatTensor).to(device)
                # batch_train_images, batch_test_images = augment_images_shapenet(batch_train_images, batch_test_images, kwargs)
            elif kwargs['dataset'] == 'bars':
                batch_train_images, batch_test_images = augment_images_bars(batch_train_images, batch_test_images,
                                                                            kwargs)
                batch_train_images = convert_channel_last_np_to_tensor(batch_train_images).to(device)
                batch_test_images = convert_channel_last_np_to_tensor(batch_test_images).to(device)
                batch_train_angles = torch.from_numpy(batch_train_angles).type(torch.FloatTensor).to(device)
                batch_test_angles = torch.from_numpy(batch_test_angles).type(torch.FloatTensor).to(device)

            # # augment shift offsets to images
            # batch_train_images = batch_train_images.reshape(-1, kwargs['img_channels'], kwargs['img_size'][0],
            #                                                 kwargs['img_size'][1])
            # batch_test_images = batch_test_images.reshape(-1, kwargs['img_channels'], kwargs['img_size'][0],
            #                                               kwargs['img_size'][1])
            # transform = RandomAffine(degrees=0, translate=(0.2, 0.2), fillcolor=0)
            #
            # batch_train_images, batch_test_images = transform(batch_train_images), transform(batch_test_images)
            # batch_train_images = batch_train_images.reshape(kwargs['tasks_per_batch'], -1,
            #                                                 kwargs['img_channels'],
            #                                                 kwargs['img_size'][0], kwargs['img_size'][1])
            # batch_test_images = batch_test_images.reshape(kwargs['tasks_per_batch'], -1, kwargs['img_channels'],
            #                                               kwargs['img_size'][0], kwargs['img_size'][1])

            all_gt_images = torch.cat((batch_train_images, batch_test_images), dim=1).cpu().detach().numpy()
            all_gt_angles = torch.cat((batch_train_angles, batch_test_angles), dim=1).cpu().detach().numpy()
            all_gt_centers = torch.cat((batch_train_centers, batch_test_centers), dim=1).cpu().detach().numpy()

            generated_centers = model(batch_test_images)
            loss = model.mse_loss(batch_test_centers, generated_centers)
            loss_all.append(loss)
            print("loss:\n", loss)
            images_to_generate = torch.cat([batch_train_images, batch_test_images], dim=1)
            centers_to_generate = torch.cat([batch_train_centers, batch_test_centers], dim=1)
            generated_centers = model(images_to_generate)
            path_save_image = os.path.join(kwargs['results_directory'], "image")
            if not os.path.exists(path_save_image):
                os.makedirs(path_save_image)
            output_path = os.path.join(path_save_image, 'output_composite_{0:02d}'.format(test_iteration))

            # generated_angles = cal_angle_from_sincos(generated_angles)
            # error = calc_error(angles_to_generate[..., 0], generated_angles)
            error = calc_distance(centers_to_generate, generated_centers)
            error_all.append(error)


            # plot_image_strips(1.0 - batch_train_images[0], generated_centers[0], 1.0 - all_gt_images[0],
            #                   all_gt_centers[0], all_gt_angles[0], data.get_image_height(), data.get_image_width(), angles_to_plot,
            #                   output_path)
            # output_folder = os.path.join(path_save_image, 'images_{0:02d}'.format(test_iteration))
            # save_images_to_folder(generated_images[0], angles_generate, all_gt_images[0], all_gt_angles[0],
            #                       output_folder)
            plot_image_and_centers(1.0 - images_to_generate[0], centers_to_generate[0], generated_centers[0], data.get_image_height(), data.get_image_width(),
                              output_path)

            test_iteration += 1

    loss_all = torch.Tensor(loss_all)
    loss_mean = torch.mean(loss_all)
    error_mean = torch.mean(torch.Tensor(error_all))

    with open(os.path.join(kwargs['results_directory'], 'results.txt'), 'w') as f:
        f.write(f"mean loss: {loss_mean.item()}")
    with open(os.path.join(kwargs['results_directory'], 'error.txt'), 'w') as f:
        f.write(f"mean error: {error_mean.item()}")


if __name__ == "__main__":
    path = kwargs["results_directory"]
    for i in range(1, kwargs['num_context_points']):
        np.random.seed(11)
        torch.manual_seed(11)
        kwargs['num_context_points'] = i
        kwargs["results_directory"] = path + f'/context_num_{i}'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)

        evaluate(device, kwargs)
