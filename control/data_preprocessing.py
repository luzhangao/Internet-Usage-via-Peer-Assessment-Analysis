# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/12/17
@description:
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils.gerenal_tools import open_yaml


paras = open_yaml("../data/samples.yaml")


def preprocessing():
    raw_data = paras["raw_path"] + paras["train_set"] + paras["train_set_file"]
    df = pd.read_csv(raw_data, sep=";")
    # print(df)
    enc = OneHotEncoder(handle_unknown='ignore')
    res = enc.fit_transform(df).toarray()
    return res


if __name__ == '__main__':
    preprocessing()
