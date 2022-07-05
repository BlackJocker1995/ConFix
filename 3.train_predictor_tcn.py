import pandas as pd

from ModelFit.approximate import CyTCN

"""
Train LSTM Model
"""
if __name__ == '__main__':
    tcn = CyTCN(100, 512)
    # read
    feature = pd.read_csv("model/features_train.csv")
    # Train
    tcn.train(feature, cuda=True)
