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

# _/_/_/_/_/_/_/ 次元削減のサンプル _/_/_/_/_/_/_/
# インクリメントPCA
# メモリに乗り切らないほどの大きいデータセットには、メモリに乗るような小さいバッチに分けて、インクリメンタルPCAを行う。

from sklearn.decomposition import IncrementalPCA

incremental_pca = IncrementalPCA(n_components=784, batch_size=None)

x_train_incremental_pca = incremental_pca.fit_transform(x_train)
x_train_incremental_pca = pd.DataFrame(data=x_train_incremental_pca, index=range(0, len(x_train)))

scatter_plot(x_train_incremental_pca, y_train, "IncrementalPCA")

# _/_/_/_/_/_/_/ 次元削減のサンプル _/_/_/_/_/_/_/
# kernel PCA
# 今までのPCAは、元のデータを低次元空間に線形に射影するが、カーネルPCAは非線形に射影する

# 類似度関数を実行し、非線形な時限削減を行う

# 類似度関数を用いる方法を総称してカーネル法と呼ぶ
from sklearn.decomposition import KernelPCA

kernel_pca = KernelPCA(n_components=100, kernel='rbf', gamma=None, n_jobs=1, random_state=2018)
kernel_pca.fit(x_train.loc[:10000, :])

x_train_kernel_pca = kernel_pca.transform(x_train)
x_train_kernel_pca = pd.DataFrame(data=x_train_kernel_pca, index=range(0, len(x_train)))
scatter_plot(x_train_kernel_pca, y_train, "Kernel PCA")

# _/_/_/_/_/_/_/ 次元削減のサンプル _/_/_/_/_/_/_/
# 特異値分解
# データの背後にある構造を学習する方法として、元の特徴量行列のランクよりも小さなランクを持つ行列を作り、小さなランクの行列のベクトルの一部の線形結合として元の行列が再構成できるようにする方法がある。この方法は、特異値分解（SVD）と呼ばれている。

# SVDは小さいランクの行列を作る際に、元の行列の最も多くの情報を持つベクトル（すなわち最も大きい特異値を持つもの）を維持する。
# 小さいランクの行列は元の特徴量空間の最も重要な要素を捉えている。

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=2018)

x_train_svd = svd.fit_transform(x_train)
x_train_svd = pd.DataFrame(data=x_train_svd, index=range(0, len(x_train)))
scatter_plot(x_train_svd, y_train, "SVD")

# _/_/_/_/_/_/_/ 次元削減のサンプル _/_/_/_/_/_/_/
# ガウス型ランダム射影
# Johnson-Lindenstraussの補題に基づく手法
# Johnson-Lindenstraussの補題とは、高次元空間を遥かに低次元空間に埋め込んだ場合、点間の距離がほぼ保存される。

# ハイパーパラメータepsで削減した特徴量空間で維持したい成分を数を指定する。
from sklearn.random_projection import GaussianRandomProjection

grp = GaussianRandomProjection(n_components='auto', eps=0.5, random_state=2018)

x_train_grp = grp.fit_transform(x_train)
x_train_grp = pd.DataFrame(data=x_train_grp, index=range(0, len(x_train)))
scatter_plot(x_train_grp, y_train, "GRP")


# _/_/_/_/_/_/_/ 次元削減のサンプル _/_/_/_/_/_/_/
# スパースランダム射影

from sklearn.random_projection import SparseRandomProjection

srp = SparseRandomProjection(n_components='auto', density='auto', eps=0.5, dense_output=False, random_state=2018)

x_train_srp = srp.fit_transform(x_train)
x_train_srp = pd.DataFrame(data=x_train_srp, index=range(0, len(x_train)))
scatter_plot(x_train_srp, y_train, "SRP")

# _/_/_/_/_/_/_/ 次元削減のサンプル _/_/_/_/_/_/_/
# Isomap

# 全ての点間の距離をユークリッド距離ではなく曲線距離もしくは測地線距離で計算を行う。
# 元の特徴量終業の低次元空間への新しい埋め込みを学習する。

from sklearn.manifold import Isomap

isomap = Isomap(n_neighbors=5, n_components=10, n_jobs=4)
isomap.fit(x_train.loc[0:5000, :])

x_train_isomap = isomap.transform(x_train)
x_train_isomap = pd.DataFrame(data=x_train_isomap, index=range(0, len(x_train)))
scatter_plot(x_train_isomap, y_train, "Isomap")

# _/_/_/_/_/_/_/ 次元削減のサンプル _/_/_/_/_/_/_/
# MDR(多次元尺度構成法)
# データポイント間の類似度を学習し、その類似度を用いて、低次元空間にモデル化する。

from sklearn.manifold import MDS

mds = MDS(n_components=2, n_init=12, max_iter=1200, metric=True, n_jobs=4, random_state=2018)

x_train_mds = mds.fit_transform(x_train.loc[0:1000, :])
x_train_mds = pd.DataFrame(data=x_train_mds, index=range(0, len(x_train))[0:1001])

scatter_plot(x_train_mds, y_train, "MDS")

# _/_/_/_/_/_/_/ 次元削減のサンプル _/_/_/_/_/_/_/
# LLE(局所線形埋め込み)

# この手法は、次元削減された空間へ写す際に局所的な近傍での距離を保つように射影する。
# データを小さい成分（観測点の近傍）に分割し、線形埋め込みとしてモデル化する。
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method='modified', random_state=2018, n_jobs=4)
lle.fit(x_train.loc[0:5000, :])

x_train_lle = lle.transform(x_train)
x_train_lle = pd.DataFrame(data=x_train_lle, index=range(0, len(x_train)))

scatter_plot(x_train_lle, y_train, "LLE")

# _/_/_/_/_/_/_/ 次元削減のサンプル _/_/_/_/_/_/_/
# t-SNE

# この手法では、類似した点は近くなり、類似してない点は遠ざけるようにする。
# 個々の高次元の点を２次元3次元空間にモデル化することで、これを実現する。

# 実用の時は、他の次元削減手法を用いてからt-SNEを用いる。（特徴量ノイズを低減すことができ、高速に実行する）

from sklearn.manifold import TSNE

t_sne = TSNE(n_components=2, learning_rate=300, perplexity=30, early_exaggeration=12, init='random', random_state=2018)

x_train_t_sne = t_sne.fit_transform(x_train_PCA.loc[:5000, :9])
x_train_t_sne = pd.DataFrame(data=x_train_t_sne, index=range(0, len(x_train))[:5001])
scatter_plot(X_train_t_sne, Y_train, "T-SNE")