from Cptool.config import toolConfig
from Cptool.mavlink import CollectMavlinkAPM

if __name__ == '__main__':
    CollectMavlinkAPM.extract_log_path(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_changed_none", threat=6)
