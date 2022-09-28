import pandas as pd

from Cptool.config import toolConfig
from ModelFit.approximate import CyTCN

"""
Train TCN Model
"""
if __name__ == '__main__':
    tcn = CyTCN(100, 512)
    # read
    feature = pd.read_csv(f"model/{toolConfig.MODE}/{toolConfig.INPUT_LEN}_{toolConfig.OUTPUT_LEN}/features_train.csv")
    # Train
    tcn.train(feature, cuda=True)
