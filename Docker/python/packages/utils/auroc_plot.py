import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import roc_curve, auc, roc_auc_score

def auROC_plot(train, predictions):
    preds = pd.concat([train, predictions.loc[:, 1]], axis=1)
    preds.columns = ['trueLabel', 'prediction']
    # auROCの計算 第一引数に正解クラス、第二引数に予測スコアのリストや配列をそれぞれ指定する
    fpr, tpr, thresholds = roc_curve(preds['trueLabel'], preds['prediction'])

    area_under_ROC = auc(fpr, tpr)

    plt.clf()
    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: Area under the curve = {0:0.2f}'.format(area_under_ROC))

    plt.legend(loc='lower right')
    plt.show()