import numpy as np
from dataset import ShapeNet3DData
from scipy.spatial.transform import Rotation as R
import sys


def main():

    data = ShapeNet3DData(path='./data/ShapeNet3D_azi360ele30',
                          img_size=[64, 64, 4],
                          train_fraction=0.8,
                          val_fraction=0.2,
                          num_instances_per_item=30,
                          seed=42,
                          aug=['task_aug'],
                          categ=['02958343'])


    for i in range(5):
        train_images, test_images, batch_train_Q, batch_test_Q = \
            data.get_batch(source='test', tasks_per_batch=10, shot=1)
        azimuth_only = True

        num_task = batch_train_Q.shape[0]
        q_train, q_test = [], []
        for i in range(num_task):
            noise_azimuth = np.random.randint(-10, 20)
            if azimuth_only:
                noise_ele = 0
            else:
                noise_ele = np.random.randint(-10, 10)
            # adapt train
            r = R.from_quat(batch_train_Q[i])
            e = r.as_euler('ZYX', degrees=True)
            old_e = e.copy()
            e[:, 0] += noise_ele
            e[:, 2] -= noise_azimuth
            q_train.append(R.from_euler('ZYX', e, degrees=True).as_quat())
            new_e = R.from_euler('ZYX', e, degrees=True).as_euler('ZYX', degrees=True)
            print(f"old_e:{old_e}, new_e:{new_e}")

        q_train = np.array(q_train)
        q_test = np.array(q_test)


if __name__ == "__main__":
    main()
