import numpy as np
import time
import kmeans
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 显示完全的结果
np.set_printoptions(threshold=1e6)

min_max_scaler = preprocessing.MinMaxScaler()

df = pd.read_csv(r'C:\Users\10257\OneDrive\桌面\data.txt', header=None)
df.drop(columns=57, inplace=True)

df_minmax = min_max_scaler.fit_transform(df)
df_scaled = preprocessing.scale(df)
df_normalized = preprocessing.normalize(df, norm='l2')
# print(df_minmax)

kmeans = KMeans(n_clusters=2)

ds = DBSCAN(eps=0.3, min_samples=2).fit(df)
kmeans.fit(df_normalized)
# 聚类情况
cluster_db = ds.labels_
cluster_k = kmeans.labels_
print(cluster_db)

# 取最后一列
cla = []
with open(r"C:\Users\10257\OneDrive\桌面\大数据作业\分类\训练集190403203211.txt") as f:
    for line in f:
        if line[-1] == '\n':
            cla.append(line[-2])
        else:
            cla.append(line[-1])
# 将str转为int
cla = list(map(int, cla))
# print(cla)

# 找两个元素中相同的元素
index = np.arange(0, 3601)
# print(len(index[cluster_db == cla]))
# k-means: 不处理：2278;  min-max归一化：2160；均值归一化：1441；L2norm：2066；L1：1902