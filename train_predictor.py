from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM
import pandas as pd
"""
Train LSTM Model
"""
if __name__ == '__main__':
    lstm = CyLSTM(100, 512)
    # read
    feature = pd.read_csv("model/features.csv")
    # Train
    lstm.train(feature, cuda=True)