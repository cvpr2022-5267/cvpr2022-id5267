import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import decomposition


class LatentVisualizer(object):
    def __init__(self, x, y, save_path, n_component=2, isSparse=False):
        self.n_comp = n_component
        # self.perplexity = 8
        self.x = x
        self.y = y
        self.isSparse = isSparse
        self.save_path = save_path

    def scale_to_normal_range(self, x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    def transform(self):
        # data = data.cpu()
        if self.isSparse:
            pca = decomposition.TruncatedSVD(n_components=15, random_state=0)
            self.x = pca.fit_transform(self.x)

        data_embedded = TSNE(n_components=self.n_comp, perplexity=10, learning_rate=200, n_iter=8000).fit_transform(self.x)

        data_embedded = self.scale_to_normal_range(data_embedded)

        if self.n_comp == 3:
            tz = data_embedded[:, 2]
            self.tz = self.scale_to_normal_range(tz)

        self.vis(data_embedded, self.y)

    def vis(self, x, y):
        # initialize a matplotlib plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # add a scatter plot with the corresponding color and label
        if self.n_comp == 2:
            scatter = ax.scatter(x[:, 0], x[:, 1], c=y, s=20, alpha=1)
            for i in range(x.shape[0]):
                ax.text(x[i, 0], x[i, 1], int(y[i]), c=plt.cm.Set1(y[i]/15))
        elif self.n_comp == 3:
            scatter = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, s=20)

        # build a legend using the labels we set previously
        # legend1 = ax.legend(*scatter.legend_elements(),
        #                     loc="upper right", title="Classes")
        # ax.add_artist(legend1)
        # ax.legend(loc='upper right')
        ax.grid(True)

        plt.savefig(f"{self.save_path}/tsne")

        # finally, show the plot
        plt.show()


if __name__ == "__main__":
    file = "../results/evaluate_plot_tsne/CNPDistractor/2021-10-15_15-34-05_distractor_datasize_None_max_maxmse_[]_seed_2578_contrastive/context_num_15/latent_all_categ.npy"
    # file = "../results/evaluate_plot_tsne/CNPDistractor/2021-10-15_15-39-07_distractor_datasize_None_max_maxmse_[]_seed_2578/context_num_15/latent_all_categ.npy"
    with open(file, "rb") as f:
        latent_z = np.load(f)
        x = latent_z[:, :-1]
        y = latent_z[:, -1]
    vis = LatentVisualizer(x, y, save_path=os.path.dirname(file))
    vis.transform()
