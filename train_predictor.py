from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM
"""
Train LSTM Model
"""
if __name__ == '__main__':
    # pd_csv = CyLSTM.merge_file_data(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/csv")
    #
    # CyLSTM.fit_trans(pd_csv)
    lstm = CyLSTM(100, 512)
    # TODO: split the Data
    feature = lstm.extract_feature(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/csv")

    lstm.train(feature, cuda=True)