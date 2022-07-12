import pandas as pd

from ModelFit.approximate import CyTCN

"""
Train LSTM Model
"""
if __name__ == '__main__':
    tcn = CyTCN(100, 512)
    # read
    feature = pd.read_csv(f"model/{toolConfig.MODE}/features_test.csv")
    # Train
    tcn.read_model()

    tcn.test(feature, cuda=True)
