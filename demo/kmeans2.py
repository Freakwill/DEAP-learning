#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.datasets import make_blobs

from kmeans import *

n_samples = 100
random_state = 170
X, _ = make_blobs(n_samples=n_samples, random_state=random_state)
K = 3

kmeans = GAKMeans(n_clusters=K, random_state=0).fit(X)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['simhei'] 
matplotlib.rcParams['font.family']='sans-serif'

y = kmeans.predict(X)
plt.scatter(X[:,0],X[:,1], c=y)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='+', c='k')
plt.title(f'GA 均值聚类 ({n_samples}个样本)')
plt.show()