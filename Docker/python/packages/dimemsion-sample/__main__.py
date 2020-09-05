import os
import time
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib import pyplot as plt
import seaborn as sns
color = sns.color_palette()

from utils.load_pkl_gz import load_pkl_gz
from utils.mnist_to_pandas import mnist_to_pandas
from utils.scatter_plot import scatter_plot

# データセットを読み込みpandasのDataFrameに変換
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file = os.path.sep.join(['', 'data', 'mnist_data', 'mnist.pkl.gz'])

train_set, validation_set, test_set = load_pkl_gz(current_path + file)

x_train, y_train = mnist_to_pandas(train_set)
x_validation, y_validation = mnist_to_pandas(validation_set, len(x_train))
x_test, y_test = mnist_to_pandas(test_set, len(x_train) + len(x_validation))


#データセットの形状の確認
print(f"Shape of x_trian:{x_train.shape}")
print(f"SHape of y_train:{y_train.shape}")
print(f"Shape of x_validation:{x_validation.shape}")
print(f"Shape of y_validation:{y_validation.shape}")
print(f"Shape of x_test:{x_test.shape}")
print(f"Shape of y_test:{y_test.shape}")

print(x_train.describe())


# _/_/_/_/_/_/_/ 次元削減のサンプル _/_/_/_/_/_/_/
# PCAを用いて次元削減を行う。
# 元の高次元データの特徴量間の相関を用いて次元を削減する

from sklearn.decomposition import PCA

# n_components=元の次元数
pca = PCA(n_components=784, whiten=False, random_state=2018)

# 訓練セットに対してPCAを行い主成分を生成する。
x_train_PCA = pca.fit_transform(x_train)
x_train_PCA = pd.DataFrame(data=x_train_PCA, index=range(0, len(x_train)))

# 各主成分の分散の割合を出力
print(f"--- explained_variance_ratio_ ---\n{pca.explained_variance_ratio_}")
print(f"Variance Explained by all 784 principal components: {sum(pca.explained_variance_ratio_)}")

importanceOfPrincipalComponents = pd.DataFrame(data=pca.explained_variance_ratio_)
importanceOfPrincipalComponents = importanceOfPrincipalComponents.T
print(importanceOfPrincipalComponents.head())

plt.clf()# 前回のグラフを消す為の処理(メモ用に)

# 主成分のグラフ表示を行う。
importance_data_view = importanceOfPrincipalComponents.T.loc[0:9, :]
importance_data_view_plot = sns.barplot(x=importance_data_view.index, y=0, data=importance_data_view)
importance_data_view_plot.get_figure().savefig("importance_data_view_plot.png")

scatter_plot(x_train_PCA, y_train, "PCA")