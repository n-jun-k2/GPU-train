import numpy as np
import pandas as pd

from sklearn.decomposition import SparsePCA

def sparse_pca_inverse_transform(sparse_pca:SparsePCA, transform_data:pd.DataFrame, data:pd.DataFrame):
    """主成分から元の次元数のデータに再構築する(SparsePCA版)

        Args:
            transform_data:SparsePCA.transform()で得られた解
            sparse_pca:SparsePCAのインスタンス

        return:
            元の次元数のデータ
    """
    return pd.DataFrame(data=np.array(transform_data).dot(sparse_pca.components_) + np.array(data.mean(axis='index')), index=data.index)
