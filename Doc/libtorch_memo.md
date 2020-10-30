# libtorch

pytorchのC++用API

[独自ライセンス。無償で使用・再配布可。](https://github.com/pytorch/pytorch/blob/master/LICENSE)
LibTorchはC++用ライブラリで、PyTorchと同等の機能がC++から利用できる。

[libtorch C++ API](https://pytorch.org/cppdocs/)

## How to build
```bash
# libtorchを使用するプロジェクトへ移動
cd LibTorchProject
# build成果物格納用ディレクトリを作成
mkdir build
# 以下ビルド手順
cmake ..
make
```
## Factory関数
| function name | overview |
| --- | --- |
| arange | 整数列を持つテンソルを返します. |
| empty | 初期化されていない値を持つテンソルを返します。 |
| eye | 単位行列を返す。 |
| full | 単一の値で満たされたテンソルを返します |
| linspace | ある間隔で線形に間隔を置いた値を持つテンソルを返します。 |
| logspace | ある間隔で対数的に間隔を空けた値を持つテンソルを返します。 |
| ones | すべてのもので満たされたテンソルを返します。 |
| rand | 一様分布から引き出された値で満たされたテンソルを返します。 |
| randint | 指定された区間からランダムに抽出された整数のテンソルを返します。 |
| randn | 単位正規分布から引き出された値で満たされたテンソルを返します。 |
| randperm |　ある間隔で整数のランダム順列で満たされたテンソルを返します。 |
| zeros | すべてゼロで満たされたテンソルを返します。 |
