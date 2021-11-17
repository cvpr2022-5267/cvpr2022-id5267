import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import PIL.Image as Image


def plot_acc(path):
    filename = os.path.join(path, "results/validate/acc.png")
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # acc
    y_cnp = np.array([0.5825, 0.775,  0.79, 0.83, 0.8275, 0.8325, 0.8425, 0.845, 0.85])
    y_bnp = np.array([0.545,  0.8325, 0.84,   0.8575, 0.8675, 0.8775, 0.885,  0.8875, 0.89])
    y_metafun = np.array([0.045,  0.64,   0.635,  0.62,   0.615,  0.6425, 0.705,  0.75,   0.765])

    unc_cnp = np.array([0.0398, 0.0432, 0.039, 0.0367, 0.0374, 0.0362, 0.0358, 0.0357, 0.035])
    unc_bnp = np.array([0.1457, 0.047, 0.0367, 0.0318, 0.0299, 0.0283, 0.0277, 0.027, 0.0266])
    unc_metafun = np.array([0.2163, 0.0234, 0.0366, 0.0496, 0.0583, 0.0611, 0.0615, 0.0596, 0.0544])
    plt.plot(x, y_cnp, label='CNP')
    plt.fill_between(x, y_cnp - unc_cnp, y_cnp + unc_cnp, alpha=0.1)
    plt.plot(x, y_bnp, label='BNP')
    plt.fill_between(x, y_bnp - unc_bnp, y_bnp + unc_bnp, alpha=0.1)
    plt.plot(x, y_metafun, label='BNP')
    plt.fill_between(x, y_bnp - unc_metafun, y_bnp + unc_metafun, alpha=0.1)
    plt.legend(loc='best')
    plt.xlabel('ctx_num')
    plt.ylabel('acc(%)')
    plt.savefig(filename)
    plt.clf()


# def plot_error(path):
#     filename = os.path.join(path, "results/validate/error.png")
#     x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#     y_cnp = np.array([2.537628, 1.469487, 1.354075, 1.180746, 1.186082, 1.176408, 1.168076, 1.149732, 1.136983])
#     y_bnp = np.array([2.59194,  1.194942, 1.116124, 1.075196, 1.026487, 1.001349, 1.001798, 0.999347, 1.005409])
#     y_metafun = np.array([7.595328, 2.455113, 2.41252,  2.354541, 2.285164, 2.099602, 1.788911, 1.637246, 1.654201])
#     unc_cnp = np.array([0.0398, 0.0432, 0.039, 0.0367, 0.0374, 0.0362, 0.0358, 0.0357, 0.035])
#     unc_bnp = np.array([0.1457, 0.047, 0.0367, 0.0318, 0.0299, 0.0283, 0.0277, 0.027, 0.0266])
#     unc_metafun = np.array([0.2163, 0.0234, 0.0366, 0.0496, 0.0583, 0.0611, 0.0615, 0.0596, 0.0544])
#     plt.plot(x, y_cnp, label='CNP')
#     plt.fill_between(x, y_cnp - unc_cnp, y_cnp + unc_cnp, alpha=0.1)
#     plt.plot(x, y_bnp, label='BNP')
#     plt.fill_between(x, y_bnp - unc_bnp, y_bnp + unc_bnp, alpha=0.1)
#     plt.plot(x, y_metafun, label='BNP')
#     plt.fill_between(x, y_bnp - unc_metafun, y_bnp + unc_metafun, alpha=0.1)
#     plt.legend(loc='best')
#     plt.xlabel('ctx_num')
#     plt.ylabel('RMS error (pixel)')
#     plt.savefig(filename)
#     plt.clf()


def plot_error():
    path = "../results/evaluate/ships_nll"
    filename = os.path.join(path, "mean_error.png")
    x = []
    error = []
    for i in range(1, 12):
        p = os.path.join(path, f"context_num_{i}")
        x.append(i)
        with open(os.path.join(p, "mean_error.txt"), "r") as f:
            e = f.readline().split(" ")[-1]
            error.append(float(e))
    plt.plot(x, error)
    plt.xlabel('ctx_num')
    plt.ylabel(r'error $(pixel)$')
    plt.savefig(filename)
    plt.clf()


def plot_all_errors():
    path = "../results/evaluate/ships_nll"
    filename = os.path.join(path, "error.png")
    x = []
    mean_error = []
    min_error = []
    for i in range(1, 12):
        p = os.path.join(path, f"context_num_{i}")
        x.append(i)
        with open(os.path.join(p, "mean_error.txt"), "r") as f:
            e = f.readline().split(" ")[-1]
            mean_error.append(float(e))
        with open(os.path.join(p, "min_error.txt"), "r") as f:
            e = f.readline().split(" ")[-1]
            min_error.append(float(e))
    plt.plot(x, mean_error, label='mean')
    plt.plot(x, min_error, label='min')
    plt.xlabel('ctx_num')
    plt.ylabel(r'error $(pixel)$')
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.clf()


def plot_all_errors_quaternion():
    path = "../results/evaluate/shapenet_6d_BACO_determ_cropped_input_quaternion_minimum_loss/2021-06-07_16-15-37_CondNeuralProcess"
    filename = os.path.join(path, "error.png")
    x = []
    mean_error = []
    folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    folders = sorted(folders, key=lambda s: int(s.split('context_num_')[1]))
    for i, f in enumerate(folders):
        error = []
        p = os.path.join(path, f)
        x.append(i + 1)
        objects = [o for o in os.listdir(p) if os.path.isdir(os.path.join(p, o))]
        # objects = ['02818832']
        for obj in objects:
            with open(os.path.join(os.path.join(p, obj), "results.txt"), "r") as f:
                e = f.readline().split(" ")[-1]
                error.append(float(e))
        mean_error.append(np.mean(error))

    plt.plot(x, mean_error, label='mean')
    plt.xlabel('ctx_num')
    plt.ylabel(r'quaternion error')
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.clf()


def plot_var():
    path = "../results/evaluate/"
    filename = os.path.join(path, "std_var.png")
    x = list(range(1, 37))
    var_gt = [ 2.2091,  0.6130,  1.3480,  0.4334,  1.6362,  1.2175,  2.2437,  3.4638,
          2.6870,  1.8059,  2.7129, 53.9661,  2.2872,  9.7120,  2.6685, 39.3095,
          0.9928,  0.1638,  1.9732,  0.7132, 13.0779,  2.6635,  1.0865,  1.1082,
          0.8812,  1.6318,  3.6587, 14.4432,  1.7035,  2.6082, 43.4758, 31.6373,
          9.7976,  0.6768,  0.6068,  8.1091]
    var_pre = [ 2.2497,  2.6529,  7.0298,  2.0906,  2.2008,  2.8361,  4.5119, 10.9757,
         5.8887,  2.3322,  4.0458, 34.8742,  2.5234, 16.5704,  2.3350, 34.9582,
         2.2697,  2.2608,  2.2083,  2.2751, 17.5449,  2.2128,  2.0478,  2.1163,
         2.8557,  2.1882,  6.1567, 18.8423,  2.3504,  2.0562, 43.3122, 36.9860,
        26.0836,  2.4179,  2.2714, 19.1257]
    plt.plot(x, var_gt, label='gt_std')
    plt.plot(x, var_pre, label='pre_std')
    plt.legend(loc='best')
    plt.xlabel('instance')
    plt.ylabel(r'std variance $(pixel)$')
    plt.savefig(filename)
    plt.clf()


def plot_uncertainty(path):
    filename = os.path.join(path, "results/validate/uncertainty.png")
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    unc_cnp = np.array([0.0398, 0.0432, 0.039, 0.0367, 0.0374, 0.0362, 0.0358, 0.0357, 0.035])
    unc_bnp = np.array([0.1457, 0.047, 0.0367, 0.0318, 0.0299, 0.0283, 0.0277, 0.027, 0.0266])
    unc_metafun = np.array([0.2163, 0.0234, 0.0366, 0.0496, 0.0583, 0.0611, 0.0615, 0.0596, 0.0544])
    plt.plot(x, unc_cnp, label='CNP')
    plt.plot(x, unc_bnp, label='BNP')
    plt.plot(x, unc_metafun, label='BNP')
    plt.legend(loc='best')
    plt.xlabel('ctx_num')
    plt.ylabel(r'uncertainty $(\delta)$')
    plt.savefig(filename)
    plt.clf()


def combine_images():
    root_path = "../results/evaluate/2021-03-18_09-13-03_CondNeuralProcess_resnet_car_32_stable"
    dir = root_path + "/combined_image"
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i in range(700):
        img_list = []
        for j in range(1, 15):
            img = Image.open(f"{root_path}/context_num_{j}/image/output_composite_{i:02d}.png").convert('RGB')
            img_list.append(np.array(img))
        img = np.stack(img_list, axis=0)
        img = img.reshape(-1, 787, 3)
        Image.fromarray(img).save(f"{dir}/output_combined_{i:02d}.png")


def plot_instances_loss_cores_to_contexts(categ):
    path = "../results/evaluate/2021-06-21_21-47-59_CondNeuralProcess_dim_256"
    filename = path + "/{}/{}/image/output_00/loss_instances.txt"
    d = np.empty([0, 14])
    folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    folders = sorted(folders, key=lambda s: int(s.split('context_num_')[1]))
    for i, f in enumerate(folders):
        p = filename.format(f, categ)
        d = np.concatenate((d, np.loadtxt(p)[:14][None, :]), axis=0)

    np.savetxt(path + '/loss_instances.txt', d, fmt='%.4f')

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    im = ax.imshow(d[:15].round(4))
    ax.set_xticks(np.arange(14))
    ax.set_yticks(np.arange(15))
    ax.set_xticklabels(np.arange(1, 15))
    ax.set_yticklabels(np.arange(15))
    ax.xaxis.tick_top()

    for i in range(15):
        for j in range(14):
            text = ax.text(j, i, d.round(4)[i, j], fontsize=8,
                           ha="center", va="center", color="w")

    ax.set_title("Quaternion loss")
    fig.tight_layout()
    plt.savefig(path + f'/loss_instances_{categ}.png')
    plt.show()
    plt.clf()


if __name__ == "__main__":
    categories = ['03928116', '02924116', '02818832']  # ['piano', 'bus', 'bed']
    p = Path(__file__)
    p = p.parent.parent
    # combine_images()
    # plot_acc(p)
    # plot_error()
    # plot_all_errors()
    # plot_all_errors_quaternion()
    for categ in categories:
        plot_instances_loss_cores_to_contexts(categ)
    # plot_var()
    # plot_uncertainty(p)
