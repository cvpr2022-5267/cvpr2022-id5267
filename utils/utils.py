import numpy as np
import random
import imgaug.augmenters as iaa
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as transforms
import torch
from collections import OrderedDict


def convert_channel_last_np_to_tensor(input):
    """input: [task_num, samples_num, height, width, channel]"""
    input = torch.from_numpy(input).type(torch.float32)
    input = input.permute(0, 1, 4, 2, 3).contiguous()
    return input

def adjust_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 0.9 every 5000 epochs"""
    lr = optimizer.param_groups[0]['lr']
    lr = lr * 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def select_from_indices(l, index):
    access_map = map(l.__getitem__, index)
    access_list = list(access_map)
    return access_list


def shuffle_img(img, label):
    """deprecated"""
    points_num = img.size(0)
    l = list(range(points_num))
    random.shuffle(l)
    img = img[l]
    label = select_from_indices(label, l)
    return img, label


def extract_img_label(img, label, color):
    """deprecated"""
    """extract img and label which contain the selected color"""
    ind = []
    for i in range(img.size(0)):
        if color in label[i]:
            ind.append(i)
    return img[ind], select_from_indices(label, ind)


def split_context_target(img, label, num_ctx, c):
    """deprecated"""
    """split data based on color/task"""
    l_ctx = []
    l_target = []
    num = img.size(0)
    if num == 0:
        return [None] * 4
    elif num <= num_ctx:
        # context number not satisfied, use last one as target
        label_ctx = label[:-1]
        for l in label_ctx:
            l_ctx.append(l[c])
        l_target.append(label[-1][c])
        return img[:-1], img[-1], l_ctx, l_target
    else:
        img_ctx = img[:num_ctx]
        img_target = img[num_ctx:]
        label_ctx = label[:num_ctx]
        for l in label_ctx:
            l_ctx.append(l[c])
        label_target = label[num_ctx:]
        for l in label_target:
            l_target.append(l[c])
    return img_ctx, img_target, l_ctx, l_target


def shuffle_data(img, label, label_c):
    num_per_task = img.size(1)
    index = torch.randperm(num_per_task)
    img = img[:, index, :, :, :]
    label = label[:, index]
    label_c = label_c[:, index, :]
    return img, label, label_c


def split_data(img, label, label_c, num_ctx):
    """
    img: [task_num, num_per_task, 3, 96, 96], bs = num_per_task * task_num
    label: [task_num, 20]
    """
    # num_per_task = img.size(1)
    # index = torch.randperm(num_per_task)
    # index_ctx = index[:num_ctx]
    # index_tst = index[num_ctx:]

    img_ctx = img[:, :num_ctx, :, :, :]
    label_ctx = label[:, :num_ctx]
    label_c_ctx = label_c[:, :num_ctx, :]

    img_tst = img[:, num_ctx:, :, :, :]
    label_tst = label[:, num_ctx:]
    label_c_tst = label_c[:, num_ctx:, :]

    # img_tst = img
    # label_tst = label
    # label_c_tst = label_c

    return img_ctx, img_tst, label_ctx, label_tst, label_c_ctx, label_c_tst


def convert_images_label(img, label, label_c, task_num):
    """
    change img to [task_num, num_per_task, c, h, w]
    img: [bs, c, h, w], bs = ctx_num * task_num
    label: [num_per_task * task_num]
    return: img [task_num, num_per_task, c, h, w]
    """
    bs, c, h, w = img.shape
    img = img.reshape(task_num, -1, c, h, w)
    label = label.reshape(task_num, -1)
    label_c = label_c.reshape(task_num, -1, 3)
    # img = img.permute(1, 0, 2, 3, 4)
    # label = label.permute(1, 0)

    return img, label, label_c


def augment_images_shapenet(batch_train_images, batch_test_images, kwargs):
    # augment shift offsets to images
    batch_train_images = batch_train_images.reshape(-1, kwargs['img_channels'], kwargs['img_size'][0],
                                                    kwargs['img_size'][1])
    batch_test_images = batch_test_images.reshape(-1, kwargs['img_channels'], kwargs['img_size'][0],
                                                  kwargs['img_size'][1])

    transform = transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), fillcolor=0)

    batch_train_images, batch_test_images = transform(batch_train_images), transform(batch_test_images)
    batch_train_images = batch_train_images.reshape(kwargs['tasks_per_batch'], -1, kwargs['img_channels'],
                                                    kwargs['img_size'][0], kwargs['img_size'][1])
    batch_test_images = batch_test_images.reshape(kwargs['tasks_per_batch'], -1, kwargs['img_channels'],
                                                  kwargs['img_size'][0], kwargs['img_size'][1])
    return batch_train_images, batch_test_images


def augment_images_bars(batch_train_images, batch_test_images, kwargs):
    # augment shift offsets to images
    batch_train_images = batch_train_images.reshape(-1, kwargs['img_size'][0],
                                                    kwargs['img_size'][1], kwargs['img_channels'])
    batch_test_images = batch_test_images.reshape(-1, kwargs['img_size'][0],
                                                  kwargs['img_size'][1], kwargs['img_channels'])

    tf = iaa.Sequential([
            iaa.flip.Fliplr(p=0.5),
            iaa.flip.Flipud(p=0.5),
            iaa.Affine(translate_percent={'x': (-0.5, 0.5)}, mode='wrap')
        ])

    batch_train_images, batch_test_images = tf(images=batch_train_images), tf(images=batch_test_images)

    batch_train_images = batch_train_images.reshape(kwargs['tasks_per_batch'], -1,
                                                    kwargs['img_size'][0], kwargs['img_size'][1], kwargs['img_channels'])
    batch_test_images = batch_test_images.reshape(kwargs['tasks_per_batch'], -1,
                                                  kwargs['img_size'][0], kwargs['img_size'][1], kwargs['img_channels'])

    return batch_train_images.astype('float32')/255.0, batch_test_images.astype('float32')/255.0


def task_augment(batch_train_Q, batch_test_Q, azimuth_only=False):
    # add random noise to each task.
    num_task = batch_train_Q.shape[0]
    q_train, q_test = [], []
    for i in range(num_task):
        noise_azimuth = np.random.randint(-10, 20)
        if azimuth_only:
            noise_ele = 0
        else:
            noise_ele = np.random.randint(-5, 10)
        # adapt train
        r = R.from_quat(batch_train_Q[i])
        e = r.as_euler('ZYX', degrees=True)
        e[:, 0] += noise_ele
        e[:, 2] -= noise_azimuth
        q_train.append(R.from_euler('ZYX', e, degrees=True).as_quat())
        # adapt test
        r = R.from_quat(batch_test_Q[i])
        e = r.as_euler('ZYX', degrees=True)
        e[:, 0] += noise_ele
        e[:, 2] -= noise_azimuth
        q_test.append(R.from_euler('ZYX', e, degrees=True).as_quat())

    q_train = np.array(q_train)
    q_test = np.array(q_test)
    return q_train, q_test


def shuffle_batch(images, R, Q, T):
    """
       Return a shuffled batch of data
    """
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], R[permutation], Q[permutation], T[permutation]


def convert_index_to_angle(index, num_instances_per_item):
    """
    Convert the index of an image to a representation of the angle
    :param index: index to be converted
    :param num_instances_per_item: number of images for each item
    :return: a biterion representation of the angle
    """
    degrees_per_increment = 360./num_instances_per_item
    angle = index * degrees_per_increment
    angle_radians = np.deg2rad(angle)
    return angle, np.sin(angle_radians), np.cos(angle_radians)


def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()


def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
            for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
            for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()


class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.

    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'

