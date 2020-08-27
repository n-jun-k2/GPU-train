import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

def train_model_get_predictions_based_on_k_folds(x_train, y_train, k_fold, model, is_debug: bool = True):
    """
    sklearnモデルの訓練

    k_fold = StratifiedKFold(n_splits=5, shuffle=True..)の場合は
    k分割交差検証を5回実行する。 
    4/5を訓練データを用い、1/5を検証データとして使って予測し結果を保持しておく。

    prams:
        x_train:訓練データ
        y_train:訓練データの解
        k_fold:訓練セットをk個に分割し検証を行う k分割交差検証オブジェクト(StratifiedKFold)
        model:sklearnモデル

    return:
        pd.DataFrame:(data=[], index=y_train.index, columns=[0,1])
    """
    cvScores = []
    trainingScores = []
    predictions_based_on_k_folds = pd.DataFrame(data=[], index=y_train.index, columns=[0,1])
    for train_index, cv_index, in k_fold.split(np.zeros(len(x_train)), y_train.ravel()):
        x_train_fold, x_cv_fold = x_train.iloc[train_index, :], x_train.iloc[cv_index, :]
        y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]

        model.fit(x_train_fold, y_train_fold)
        log_loss_training = log_loss(y_train_fold, model.predict_proba(x_train_fold)[:, ])
        trainingScores.append(log_loss_training)

        predictions_based_on_k_folds.loc[x_cv_fold.index, :] = model.predict_proba(x_cv_fold)

        y_pred = predictions_based_on_k_folds.loc[x_cv_fold.index, 1]
        log_loss_cv = log_loss(y_cv_fold, y_pred)

        cvScores.append(log_loss_cv)
        if is_debug:
            print('Training Log Loss: ', log_loss_training)
            print('CV Log Loss: ', log_loss_cv)
    return predictions_based_on_k_folds