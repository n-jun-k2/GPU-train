# 次元削減＝情報を密に捉える手法
# クラスタリング=類似性に基づいて観測点をグループ分けする手法

import numpy as np
import pandas as pd
import os, time

# 一つ上の階層を探索範囲に加える。
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.load_pkl_gz import load_pkl_gz
from utils.mnist_to_pandas import mnist_to_pandas

from sklearn.decomposition import PCA

current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file = os.path.sep.join(['', 'data', 'mnist_data', 'mnist.pkl.gz'])

train_set, validation_set, test_set = load_pkl_gz(current_path + file)

x_train, y_train = mnist_to_pandas(train_set)
x_validation, y_validation = mnist_to_pandas(validation_set, len(x_train))
x_test, y_test = mnist_to_pandas(test_set, len(x_train) + len(x_validation))

# クラスタリングを行う前にPCAを用いて次元削減を行う。
pca = PCA(n_components=784, whiten=False, random_state=2018)
x_train_pca = pca.fit_transform(x_train)
x_train_pca = pd.DataFrame(data=x_train_pca, index=x_train.index)

# このタイミングでは次元削減を行わないが、クラスタリングを行う際に利用する主成分の数を指定する際に次元数を減らすことで次元削減と同じ効果が得られる。
""" クラスタリング
K平均法、階層クラスタリング、DBSCAN

目的：データセットを次の性質を満たすようなグループに分割すること

"""

"""
k平均法：
クラスタ数ｋを指定し
個々の観測点にただ一つだけクラスタを割り当てる為クラスタ内ばらつきもしくはクラスタ慣性を最小化する。

クラスタリングの開始位置はランダムで行う。
その後、クラスタの中心点と各観測点のユークリッド距離が最小となるクラスタに割り当てなおす。
初期化がランダムで行われる為実行するたび結果がわずかに変わる。
"""
from sklearn.cluster import KMeans

n_clusters = 10
kMeans_inertia = pd.DataFrame(data=[], index=range(2,21), columns=['inertia'])

for n_clusters in range(2, 21):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, tol=0.0001, random_state=2018)
    kmeans.fit(x_train_pca.loc[:, 0:99])
    kMeans_inertia.loc[n_clusters] = kmeans.inertia_

# クラスタリングの場合次元数を任意に決めることが出来る為以下の様にグラフ出力し生成するべき適切なクラスタ数を見つけること。
#sns.set()
#sns.lineplot(data=kMeans_inertia, x=kMeans_inertia.index, y='inertia')
#plt.show()