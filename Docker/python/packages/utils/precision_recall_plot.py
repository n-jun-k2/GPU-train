import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import precision_recall_curve, average_precision_score

def precision_recall_plot(train, predictions):
    """
    評価尺度
    適合率 = 陽性と予測された真陽性の割合　真陽性/(真陽性＋偽陽性)
    再現率 = データセット中の実際に陽性の内、陽性だと予測された割合。真陽性/(真陽性＋偽陽性)
            適合率 < 再現率 = 誤報が多い。
    適合率 > 再現率 = 陽性として認識するのが少ないが実際に陽性である場合が多い。
            特異度 = 真陰性率 = 真陰性 / (真陰性 + 偽陰性)
    """
    preds = pd.concat([train, predictions.loc[:, 1]], axis=1)
    preds.columns = ['trueLabel', 'prediction']
    
    #異なる確率しきい値の精度と再現率のペアを計算
    precision, recall, _ = precision_recall_curve(preds['trueLabel'], preds['prediction'])

    average_precision = average_precision_score(preds['trueLabel'], preds['prediction'])

    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))