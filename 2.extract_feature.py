from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM, CyTCN

"""
Train LSTM Model
"""
if __name__ == '__main__':
    # pd_csv = CyLSTM.merge_file_data(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_changed/csv")

    # CyLSTM.fit_trans(pd_csv)
    tcn = CyTCN(100, 512)
    feature = tcn.extract_feature(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_changed/csv")
    # Save
    feature.to_csv(f"model/{toolConfig.MODE}/{toolConfig.INPUT_LEN}_{toolConfig.OUTPUT_LEN}/features.csv", index=False)
