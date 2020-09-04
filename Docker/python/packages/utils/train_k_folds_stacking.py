import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .train_k_folds import train_model_get_predictions_based_on_k_folds

def train_model_get_predictions_based_on_k_folds_stacking(x_train, y_train, k_fold, model_dict: dict, select_train_key: str) -> pd.DataFrame :
    """
    スタッキング：アンサンブル手法の一種

    1. 各モデルから予測値を算出<layer1>
    2. layer1で取得した値を訓練セットに追加する
    3. 2の訓練セットで各モデルの予測値を算出<layer2>


    param:
        x_train:訓練データ
        y_train:訓練データの解
        k_fold:訓練セットをk個に分割し検証を行う　k分割交差検証オブジェクト（StratifiedKFold）
        model_dict:{モデル名：sklearnモデル, ...}実行するモデル辞書

    """
    predictions_based_on_k_folds = pd.DataFrame(data=[], index=y_train.index)
    # layer1を求める。
    layer1_prediction_list = [ train_model_get_predictions_based_on_k_folds(x_train, y_train, k_fold, model, False)for model in model_dict.values()]

    # 予測値の行列を作成
    for index, layer1 in enumerate(layer1_prediction_list):
        predictions_based_on_k_folds = predictions_based_on_k_folds.join(layer1[1].astype(float), how='left', rsuffix=str(index))

    # 列名をkey名に統一
    predictions_based_on_k_folds.columns = list(model_dict.keys())

    # 訓練セットと予測値を結合する
    x_train_with_predictions = x_train.merge(predictions_based_on_k_folds, left_index=True, right_index=True)

    # チューニング対象のモデルをキャッシュ
    train_model = model_dict[select_train_key]

    # layer2を求める。
    layer2_prediction_based_on_k_fold = train_model_get_predictions_based_on_k_folds(x_train, y_train, k_fold, train_model, False)

    return layer2_prediction_based_on_k_fold