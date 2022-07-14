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
    feature = pd.read_csv(f"model/{toolConfig.MODE}/features_train.csv")
    # Train
    tcn.train(feature, cuda=True)
