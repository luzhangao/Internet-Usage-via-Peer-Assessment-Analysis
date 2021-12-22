# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/12/17
@description:
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from utils.gerenal_tools import open_yaml, save_pickle


paras = open_yaml("../data/samples.yaml")


def preprocessing():
    raw_data = paras["raw_path"] + paras["train_path"] + paras["train_file"]
    # df = pd.read_csv(raw_data, sep=";")
    df = pd.read_excel(raw_data)
    # Remove all data from respondents: parents (Dr.K)
    df = df[df["Type of person"] != "parents"]
    # Fill nan with mean
    df = df.fillna(df.mean(numeric_only=True))
    print(df)
    # Split the dataframe
    cdf = df[paras["continuous_columns"]]
    ddf = df[paras["discrete_columns"]]

    scaler = MinMaxScaler()
    carray = scaler.fit_transform(cdf)
    c_feature_names = scaler.get_feature_names_out()

    enc = OneHotEncoder(handle_unknown='ignore')
    darray = enc.fit_transform(ddf).toarray()
    d_feature_names = enc.get_feature_names_out()

    # print(carray.shape, darray.shape)
    res = np.concatenate([carray, darray], axis=1)
    feature_names = np.concatenate([c_feature_names, d_feature_names])
    # print(res.shape)
    # print(feature_names)
    np.save(paras["raw_path"] + paras["train_path"] + paras["train_dataset"], res)
    save_pickle(feature_names, paras["model_saved_path"] + paras["feature_names"])
    return res


if __name__ == '__main__':
    preprocessing()
