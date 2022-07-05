import pandas as pd

from ModelFit.approximate import CyLSTM

"""
Train LSTM Model
"""
if __name__ == '__main__':
    lstm = CyLSTM(100, 512)
    # read
    feature = pd.read_csv("model/features_train.csv")
    # Train
    lstm.train(feature, cuda=True)
