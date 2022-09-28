import pandas as pd

from Cptool.config import toolConfig
from ModelFit.approximate import CyTCN

"""
Train LSTM Model
"""
if __name__ == '__main__':
    toolConfig.select_mode("PX4")

    tcn = CyTCN(100, 512)
    # read
    feature = pd.read_csv(f"model/{toolConfig.MODE}/{toolConfig.INPUT_LEN}_{toolConfig.OUTPUT_LEN}/features_test.csv")
    # Train
    tcn.read_model()

    tcn.test(feature, cuda=True)
