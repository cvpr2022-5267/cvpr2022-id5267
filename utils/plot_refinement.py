import numpy as np
import matplotlib.pyplot as plt


def plot_distractor_task():
    path = "/home/ning/toy_task/run_experiments/results/eval_one_task/CNPDistractor/2021-10-04_08-58-09_distractor_datasize_None_max_maxmse_['data_aug']_seed_2578"
    filename = f"{path}/refine_distractor.png"
    test_losses = np.loadtxt(f"{path}/test_losses.txt")
    test_losses = test_losses[:, 1]
    index = list(range(1, 25 + 1))
    test_losses = np.array(test_losses)
    # test_std = np.array(test_std)

    plt.rcParams.update({'font.size': 17})

    plt.plot(index, test_losses, label='CNP')
    # plt.fill_between(index, test_losses - test_std, test_losses + test_std, alpha=0.1)

    path = "/home/ning/toy_task/run_experiments/results/refinement/SingleTaskDistractor/2021-10-03_22-47-35_distractor_datasize_None__maxmse_['data_aug']_seed_2578"
    test_losses = np.loadtxt(f"{path}/loss_vs_ctx.txt")
    test_losses = test_losses[:]
    index = list(range(1, 25 + 1))
    test_losses = np.array(test_losses)
    plt.plot(index, test_losses, label='Finetuned model')

    plt.legend(loc='best')
    plt.xlabel('ctx_num')
    plt.ylabel(r'error ($px$)')
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


def plot_shapenet1d_task():
    path = "/home/ning/toy_task/run_experiments/results/eval_one_task/ANPShapeNet1D/2021-10-03_22-30-41_shapenet_1d_datasize_large_attention_mse_['task_aug', 'data_aug']_seed_2578"
    filename = f"{path}/finetune_shapenet1d.png"
    test_losses = np.loadtxt(f"{path}/test_losses.txt")
    test_losses = test_losses[:, 1]
    index = list(range(1, 25 + 1))
    test_losses = np.array(test_losses)
    # test_std = np.array(test_std)

    plt.rcParams.update({'font.size': 17})

    plt.plot(index, test_losses, label='ANP')
    # plt.fill_between(index, test_losses - test_std, test_losses + test_std, alpha=0.1)

    path = "/home/ning/toy_task/run_experiments/results/refinement/SingleTaskShapeNet1D/2021-10-01_12-50-09_shapenet_1d_datasize_large__mse_['data_aug']_seed_2578"
    test_losses = np.loadtxt(f"{path}/loss_vs_ctx.txt")
    test_losses = test_losses[:]
    index = list(range(1, 25 + 1))
    test_losses = np.array(test_losses)
    plt.plot(index, test_losses, label='Finetuned model')

    plt.legend(loc='best')
    plt.xlabel('ctx_num')
    plt.ylabel(r'error ($^\circ$)')
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    plot_shapenet1d_task()
