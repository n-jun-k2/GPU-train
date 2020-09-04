import pandas as pd


def mnist_to_pandas(dataset: tuple, index_offset: int = 0):
    """Convert mnist dataset to dataframe
    """
    data, label = dataset[0], dataset[1]
    index = range(index_offset, index_offset + len(data))

    return pd.DataFrame(data=data, index=index), pd.Series(data=label, index=index)