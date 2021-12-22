# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/12/18
@description:
"""

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from control.analyze import clustering
from utils.gerenal_tools import open_pickle, open_yaml


paras = open_yaml("../data/samples.yaml")


def weight_importance():
    temp = clustering()
    X = temp["train_dataset"]
    y = temp["labels"]
    print(X.shape, y.shape)
    model = xgb.XGBClassifier()
    model.fit(X, y)
    # y_pred = model.predict(X)
    # accuracy = accuracy_score(y, y_pred)
    # print(accuracy)
    feature_names = open_pickle(paras["model_saved_path"] + paras["feature_names"])
    print(feature_names)
    model.get_booster().feature_names = list(feature_names)
    xgb.plot_importance(model)
    plt.show()


if __name__ == '__main__':
    weight_importance()

