import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

def mnist_plot(image_df: pd.DataFrame, label_series: pd.Series, example: int):
    label = label_series.loc[example]
    image = image_df.loc[example, :].values.reshape([28,28])
    plt.title('Example: %d Label: %d' % (example, label))
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()