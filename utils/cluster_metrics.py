import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score, mutual_info_score, calinski_harabasz_score, davies_bouldin_score


class ClusterMetrics(object):
    def __init__(self, x, y, n_component=2, isSparse=True, n_clusters=12):
        self.x = x
        self.y = y
        self.n_comp = n_component
        self.isSparse = isSparse
        self.n_clusters = n_clusters

    def scale_to_normal_range(self, x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    def silhouette_score(self,):
        self.x = self.transform()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(self.x) + (self.n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=self.n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(self.x)
        # cluster_labels = self.y

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(self.x, cluster_labels)
        print(
            "For n_clusters =",
            self.n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(self.x, cluster_labels)

        y_lower = 10
        for i in range(self.n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / self.n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / self.n_clusters)
        ax2.scatter(
            self.x[:, 0], self.x[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % self.n_clusters,
            fontsize=14,
            fontweight="bold",
        )

        # plt.show()

    def adjusted_rand_score(self):
        self.x = self.transform()
        clusterer = KMeans(n_clusters=self.n_clusters, random_state=10)
        y_pr = clusterer.fit_predict(self.x)
        y_gt = self.y
        score = adjusted_rand_score(y_gt, y_pr)
        print(f"adjusted_rand_score:{score}")

    def mutual_info_score(self):
        self.x = self.transform()
        clusterer = KMeans(n_clusters=self.n_clusters, random_state=10)
        y_pr = clusterer.fit_predict(self.x)
        y_gt = self.y
        score = mutual_info_score(y_gt, y_pr)
        print(f"cluster={self.n_clusters}, mutual_info_score:{score}")

    def calinski_harabasz_score(self):
        # self.x = self.transform()
        clusterer = KMeans(n_clusters=self.n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(self.x)
        score = calinski_harabasz_score(self.x, cluster_labels)
        print(f"cluster={self.n_clusters}, calinski_harabasz_score:{score}")

    def davies_bouldin_score(self):
        # self.x = self.transform()
        clusterer = KMeans(n_clusters=self.n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(self.x)
        score = davies_bouldin_score(self.x, cluster_labels)
        print(f"cluster={self.n_clusters}, davies_bouldin_score:{score}")

    def transform(self):
        # data = data.cpu()
        if self.isSparse:
            pca = decomposition.TruncatedSVD(n_components=30, random_state=0)
            self.x = pca.fit_transform(self.x)

        data_embedded = TSNE(n_components=self.n_comp, perplexity=30, learning_rate=300, n_iter=3000).fit_transform(self.x)

        data_embedded = self.scale_to_normal_range(data_embedded)

        if self.n_comp == 3:
            tz = data_embedded[:, 2]
            self.tz = self.scale_to_normal_range(tz)

        # self.vis(data_embedded, self.y)
        return data_embedded

    def vis(self, x, y):
        # initialize a matplotlib plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # add a scatter plot with the corresponding color and label
        if self.n_comp == 2:
            scatter = ax.scatter(x[:, 0], x[:, 1], c=y, s=20, alpha=1)
            # for i in range(x.shape[0]):
            #     ax.text(x[i, 0], x[i, 1], int(y[i]), c=plt.cm.Set1(y[i]/15))
        elif self.n_comp == 3:
            scatter = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, s=20)

        # build a legend using the labels we set previously
        # legend1 = ax.legend(*scatter.legend_elements(),
        #                     loc="upper right", title="Classes")
        # ax.add_artist(legend1)
        # ax.legend(loc='upper right')
        ax.grid(True)

        # finally, show the plot
        plt.show()


if __name__ == "__main__":
    # file = "../results/evaluate_plot_tsne/CNPDistractor/2021-10-15_15-34-05_distractor_datasize_None_max_maxmse_[]_seed_2578_contrastive/context_num_15/latent_all_categ.npy"
    file = "../results/evaluate_plot_tsne/CNPDistractor/2021-10-15_15-39-07_distractor_datasize_None_max_maxmse_[]_seed_2578/context_num_15/latent_all_categ.npy"
    with open(file, "rb") as f:
        latent_z = np.load(f)
        x = latent_z[:, :-1]
        y = latent_z[:, -1]


    for c in range(12, 12+1):
        vis = ClusterMetrics(x, y)
        vis.n_clusters = c
        vis.adjusted_rand_score()
        # vis.silhouette_score()
        # vis.mutual_info_score()
        # vis.calinski_harabasz_score()
        # vis.davies_bouldin_score()
