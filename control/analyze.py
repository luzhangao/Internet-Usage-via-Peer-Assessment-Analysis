# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/12/17
@description:
"""

import numpy as np
import pandas as pd
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
from utils.gerenal_tools import open_yaml, save_pickle, open_pickle


paras = open_yaml("../data/samples.yaml")


def clustering(metric="euclidean", graph=False):
    """

    :param metric:
    :param graph:
    :return:
    """
    X = np.load(paras["raw_path"] + paras["train_path"] + paras["train_dataset"])
    # print(X)
    cm = KMeans(n_clusters=2).fit(X)
    # cm = AffinityPropagation(random_state=5).fit(X)
    # cm = MeanShift(bandwidth=2.5).fit(X)
    # cm = SpectralClustering(n_clusters=5, assign_labels='discretize', random_state=0).fit(X)
    # cm = AgglomerativeClustering(affinity=metric, linkage="average").fit(X)  # average/ward
    # cm = DBSCAN(eps=1, min_samples=20).fit(X)
    y = cm.labels_
    # y = np.array([1 - i for i in y])
    # print(y)

    # Save the results for further analysis.
    df = open_pickle(paras["raw_path"] + paras["train_path"] + paras["raw_data_file"])
    df["predict_label"] = y
    print(df)
    df.to_excel(paras["raw_path"] + paras["temp_file"])

    if graph:
        sns.set_style('darkgrid')
        sns.set_palette('muted')
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

        # dd = TSNE(random_state=0, n_components=2, metric=metric).fit_transform(X)
        dd = TSNE(random_state=0, n_components=2).fit_transform(X)
        scatter(dd, y)
        plt.show()

    ss = silhouette_score(X, y, metric=metric)  # Silhouette Coefficient, [-1, 1], greater is better.
    ch = calinski_harabasz_score(X, y)  # CH index, greater is better.
    db = davies_bouldin_score(X, y)  # DBI, less is better.
    print("metric: {metric}\nSilhouette Coefficient: {ss}\nCH index: {ch}\nDBI: {db}"
          .format(metric=metric, ss=ss, ch=ch, db=db))

    return {"metric": metric, "ss": ss, "ch": ch, "db": db, "train_dataset": X, "labels": y}


def scatter(x, colors):
    palette = np.array(sns.color_palette("hls", 6))

    f = plt.figure(figsize=(8, 8))
    # ax = plt.axes(projection="3d")
    # sc = ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], lw=0, s=40, c=palette[colors.astype(np.int)])
    ax = plt.axes()
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    # ax.axis('off')
    # ax.axis('tight')

    return f, ax, sc, [0, 1]


def run():
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
    df = pd.DataFrame({"Silhouette Coefficient": sss, "Calinski-Harabasz Index": chs, "Davies Bouldin Score": dbs})
    df = df.applymap(lambda x: round(x, 4))
    print(df)
    df.to_csv(paras["raw_path"] + paras["train_path"] + "intrinsic_metrics_of_clustering.csv")
    print(sorted(sss.items(), key=lambda item: item[1], reverse=True))
    print(sorted(chs.items(), key=lambda item: item[1], reverse=True))
    print(sorted(dbs.items(), key=lambda item: item[1]))
    """
    [('yule', 0.16541165363689975), ('correlation', 0.08996816407584957), ('cosine', 0.08754904889327006), ('cityblock', 0.08205540082317561), ('sqeuclidean', 0.08205540082317561), ('canberra', 0.08205540082317561), ('hamming', 0.07312980311971097), ('matching', 0.07312980311971097), ('braycurtis', 0.07066540166698916), ('dice', 0.07066540166698916), ('rogerstanimoto', 0.06398812656748301), ('sokalmichener', 0.06398812656748301), ('jaccard', 0.055954158770970125), ('euclidean', 0.0450879402948254), ('minkowski', 0.0450879402948254), ('sokalsneath', 0.030645091957197765), ('seuclidean', 0.025075451408111054), ('russellrao', 0.0194645244351754), ('kulsinski', 0.014695369114768549), ('chebyshev', 4.819277108433795e-05)]
    [('jaccard', 24.521762199327103), ('russellrao', 24.35660313809395), ('correlation', 24.20168099388329), ('euclidean', 24.155624642273562), ('minkowski', 24.155624642273562), ('kulsinski', 24.155624642273562), ('rogerstanimoto', 24.155624642273562), ('sokalmichener', 24.155624642273562), ('cosine', 22.98228797650974), ('sokalsneath', 22.924972407570134), ('cityblock', 22.91646894776369), ('sqeuclidean', 22.91646894776369), ('canberra', 22.91646894776369), ('yule', 22.099512601765632), ('hamming', 19.343675851590003), ('matching', 19.343675851590003), ('braycurtis', 18.195466213233786), ('dice', 18.195466213233786), ('seuclidean', 12.223736493914538), ('chebyshev', 0.9771892802983222)]
    [('chebyshev', 1.0093828432969085), ('correlation', 4.339322511437038), ('cosine', 4.383933631893184), ('russellrao', 4.3887587732246525), ('jaccard', 4.400795973255812), ('euclidean', 4.532878331745464), ('minkowski', 4.532878331745464), ('kulsinski', 4.532878331745464), ('rogerstanimoto', 4.532878331745464), ('sokalmichener', 4.532878331745464), ('sokalsneath', 4.571477674641366), ('cityblock', 4.6273712014838475), ('sqeuclidean', 4.6273712014838475), ('canberra', 4.6273712014838475), ('yule', 4.726442376677368), ('braycurtis', 4.845501291712349), ('dice', 4.845501291712349), ('hamming', 4.855974043657837), ('matching', 4.855974043657837), ('seuclidean', 5.244403669628422)]
    Correlation is the best choice. 
    """


if __name__ == '__main__':
    # clustering("correlation", graph=True)
    clustering(graph=True)
    # run()
