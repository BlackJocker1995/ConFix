import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Cptool.config import toolConfig
from ModelFit.approximate import Modeling, CyTCN


def reject_outliers(data, m=2):
    """
    three sigma filter
    :param data:
    :param m:
    :return:
    """
    return data[abs(data - np.mean(data)) < m * np.std(data)]


if __name__ == '__main__':
    cylstm = CyTCN(100, 128)
    feature = pd.read_csv(f"model/{toolConfig.MODE}/{toolConfig.INPUT_LEN}_{toolConfig.OUTPUT_LEN}/features_change.csv")
    cylstm.read_model()

    feature_x, feature_y = cylstm.data_split(feature)
    if isinstance(cylstm, CyTCN):
        feature_y = feature_y.reshape((feature_y.shape[0], -1))

    predicted_feature = cylstm.predict_feature(feature_x)
    if isinstance(cylstm, CyTCN):
        predicted_feature = predicted_feature.reshape((predicted_feature.shape[0], -1))

    patch_array_loss = Modeling.cal_patch_deviation(feature_y, predicted_feature)

    # patch_array_loss = CyLSTM.loss_discriminate(status_deviation)

    patch_array_loss = reject_outliers(patch_array_loss, 3)

    _ = plt.hist(patch_array_loss, bins='auto')

    plt.show()

    print(f"Max: {patch_array_loss.max()}  Min:{patch_array_loss.min()} AVG:{np.average(patch_array_loss)}")
