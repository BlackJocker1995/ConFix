import pandas as pd

from ModelFit.approximate import CyLSTM

"""
Train LSTM Model
"""
if __name__ == '__main__':
    lstm = CyLSTM(100, 512)
    # read
    feature = pd.read_csv(f"model/{toolConfig.MODE}/features_test.csv")
    # Train
    lstm.read_model()

    lstm.test(feature, cuda=True)
