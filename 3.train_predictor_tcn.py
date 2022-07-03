from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM, CyTCN
import pandas as pd
"""
Train LSTM Model
"""
if __name__ == '__main__':
    tcn = CyTCN(100, 512)
    # read
    feature = pd.read_csv("model/features_train.csv")
    # Train
    tcn.train(feature, cuda=True)