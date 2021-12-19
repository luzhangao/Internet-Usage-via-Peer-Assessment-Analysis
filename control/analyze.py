# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/12/17
@description:
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils.gerenal_tools import open_yaml


paras = open_yaml("../data/samples.yaml")


def clustering(metric="euclidean", graph=False):
    """

    :param metric:
    :param graph:
    :return:
    """
    X = np.load(paras["raw_path"] + paras["train_path"] + paras["train_dataset"])
    # print(X)
    # cm = KMeans(n_clusters=2).fit(X)
    # cm = AffinityPropagation(random_state=5).fit(X)
    # cm = MeanShift(bandwidth=2.5).fit(X)
    # cm = SpectralClustering(n_clusters=5, assign_labels='discretize', random_state=0).fit(X)
    cm = AgglomerativeClustering(affinity=metric, linkage="average").fit(X)  # average/ward
    # cm = DBSCAN(eps=1.5, min_samples=2).fit(X)
    y = cm.labels_
    # y = np.array([1 - i for i in y])
    # print(y)

    if graph:
        sns.set_style('darkgrid')
        sns.set_palette('muted')
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

        dd = TSNE(random_state=0, n_components=3, metric=metric).fit_transform(X)
        scatter(dd, y)
        plt.show()

    ss = silhouette_score(X, y, metric=metric)  # Silhouette Coefficient, [-1, 1], greater is better.
    ch = calinski_harabasz_score(X, y)  # CH index, greater is better.
    db = davies_bouldin_score(X, y)  # DBI, less is better.
    print("metric: {metric}\nSilhouette Coefficient: {ss}\nCH index: {ch}\nDBI: {db}"
          .format(metric=metric, ss=ss, ch=ch, db=db))
    return {"metric": metric, "ss": ss, "ch": ch, "db": db}


def scatter(x, colors):
    palette = np.array(sns.color_palette("hls", 5))

    f = plt.figure(figsize=(8, 8))
    # ax = plt.axes(projection="3d")
    # sc = ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], lw=0, s=40, c=palette[colors.astype(np.int)])
    ax = plt.axes()
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    # ax.axis('off')
    # ax.axis('tight')

    return f, ax, sc, [0, 1]


def run():
    # correlation, hamming, yule, russellrao,
    metrics = ["euclidean", "minkowski", "cityblock", "seuclidean", "sqeuclidean", "cosine", "correlation", "hamming",
               "jaccard", "chebyshev", "canberra", "braycurtis", "yule", "matching", "dice", "kulsinski",
               "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath"]
    sss = {}
    chs = {}
    dbs = {}
    for metric in metrics:
        temp = clustering(metric)
        ss = temp["ss"]
        ch = temp["ch"]
        db = temp["db"]
        sss[metric] = ss
        chs[metric] = ch
        dbs[metric] = db
    print(sorted(sss.items(), key=lambda item: item[1], reverse=True))
    print(sorted(chs.items(), key=lambda item: item[1], reverse=True))
    print(sorted(dbs.items(), key=lambda item: item[1]))

    # silhouette_score, calinski_harabasz_score, davies_bouldin_score


if __name__ == '__main__':
    # clustering("correlation", graph=True)
    run()
