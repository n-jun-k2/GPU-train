import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import matplotlib as mpl


def scatter_plot(x_dataframe: pd.DataFrame, y_dataframe: pd.Series, algo_name: str):
    """
    PCAから得られたデータを散布図を用いて可視化。

    param:
        x_dataframe：PCAから得られた訓練データ
        y_dataframe：ラベルデータ
        algo_name：アルゴリズムの名前（グラフのタイトルになる)

    """
    temp_df = pd.DataFrame(data=x_dataframe.loc[:, 0:1], index=x_dataframe.index)
    temp_df = pd.concat((temp_df, y_dataframe), axis=1, join="inner")
    temp_df.columns = ["First Vector", "Second Vector", "Label"]
    sns.lmplot(x="First Vector", y="Second Vector", hue="Label", data=temp_df, fit_reg=False)

    ax = plt.gca()
    ax.set_title("Separation of Observations usin " + algo_name)