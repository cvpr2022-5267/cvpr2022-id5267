from PIL import Image
import torch
from torchvision.transforms import transforms
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
from dataset.shapenet_distractor import ShapeNetData
from dataset.bars import Bars
from train_shapenet import convert_channel_last_np_to_tensor, augment_images_shapenet, augment_images_bars


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {
    "method": "CondNeuralProcess",
    "dataset": "bars",  # "bars" or "shapenet"
    "which_aggregate": "max",
    "img_size": [128, 128],
    "tasks_per_batch": 1,  # number of minibatch
    "num_context_points": 1,  # Number of context points
    "device": str(device),
}


if __name__ == "__main__":
    # load dataset
    if kwargs['dataset'] == 'shapenet':
        data = ShapeNetData(path='./data',
                            img_size=kwargs['img_size'],
                            train_fraction=0.7,
                            val_fraction=0.1,
                            num_instances_per_item=36,
                            seed=42,
                            mode='train')
        kwargs['label_dim'] = 3
    elif kwargs['dataset'] == 'bars':
        data = Bars(path='./data',
                    img_size=kwargs['img_size'],
                    train_fraction=0.7,
                    val_fraction=0.1,
                    num_instances_per_item=30,
                    seed=42,
                    mode='train')
        kwargs['label_dim'] = 1
    kwargs['img_channels'] = data.get_image_channels()

    batch_train_images, batch_test_images, batch_train_angles, batch_test_angles = \
        data.get_batch(source='train', tasks_per_batch=kwargs['tasks_per_batch'], shot=kwargs['num_context_points'])
    img_before = batch_train_images.squeeze()
    img_before = Image.fromarray(img_before)
    img_before.show()
    if kwargs['dataset'] == 'shapenet':
        batch_train_images = convert_channel_last_np_to_tensor(batch_train_images).to(device)
        batch_test_images = convert_channel_last_np_to_tensor(batch_test_images).to(device)
        batch_train_angles = torch.from_numpy(batch_train_angles).type(torch.FloatTensor).to(device)
        batch_test_angles = torch.from_numpy(batch_test_angles).type(torch.FloatTensor).to(device)
        batch_train_images, batch_test_images = augment_images_shapenet(batch_train_images, batch_test_images, kwargs)
    elif kwargs['dataset'] == 'bars':
        batch_train_images, batch_test_images = augment_images_bars(batch_train_images, batch_test_images, kwargs)
        batch_train_images = convert_channel_last_np_to_tensor(batch_train_images).to(device)
        batch_test_images = convert_channel_last_np_to_tensor(batch_test_images).to(device)
        batch_train_angles = torch.from_numpy(batch_train_angles).type(torch.FloatTensor).to(device)
        batch_test_angles = torch.from_numpy(batch_test_angles).type(torch.FloatTensor).to(device)

    trans_to_PIL = transforms.ToPILImage()
    img_after = batch_train_images.squeeze()
    img_after = trans_to_PIL(img_after)
    img_after.show()

    # plt.imshow(tensor_t_numpy)
    # plt.show()
    # plt.clf()
