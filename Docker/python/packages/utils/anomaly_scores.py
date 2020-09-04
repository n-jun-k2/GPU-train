import numpy as np
import pandas as pd

def anormaly_scores(original_df: pd.DataFrame, reduced_df: pd.DataFrame):
    """異常スコア関数

        Args:
            - original_df:元の特徴量ベクトル
            - reduced_df:次元削減されたベクトル

        return:
            - loss:再構成誤差（0 ~ 1）
    """
    loss = np.sum( (np.array(original_df) - np.array(reduced_df))**2, axis=1)
    loss = pd.Series(data=loss, index=original_df.index)
    loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))
    return loss