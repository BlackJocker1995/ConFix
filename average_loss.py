import numpy as np
import pandas as pd

from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM


def reject_outliers(data, m=2):
    """
    three sigma filter
    :param data:
    :param m:
    :return:
    """
    return data[abs(data - np.mean(data)) < m * np.std(data)]


if __name__ == '__main__':
    cylstm = CyLSTM(100, 128)
    feature = cylstm.extract_feature(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/csv")
    cylstm.read_model()

    feature_x, feature_y = cylstm.data_split(feature)

    predicted_feature = cylstm.predict_feature(feature_x)

    status_deviation = np.abs(feature_y - predicted_feature)

    patch_array_loss = CyLSTM.loss_discriminate(status_deviation)

    patch_array_loss = reject_outliers(patch_array_loss, 3)

    print(patch_array_loss)