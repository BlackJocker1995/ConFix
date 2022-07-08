from Cptool.config import toolConfig
from Cptool.mavlink import FixMavlinkAPM

if __name__ == '__main__':
    FixMavlinkAPM.extract_log_path(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_ordinary", threat=6)
