from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM
import pandas as pd
"""
Train LSTM Model
"""
if __name__ == '__main__':
    lstm = CyLSTM(100, 512)
    feature = lstm.extract_feature(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_ordinary/csv")
    # Save
    feature.to_csv("model/features_ordinary.csv", index=False)