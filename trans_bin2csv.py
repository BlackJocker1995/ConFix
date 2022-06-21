from Cptool.config import toolConfig
from Cptool.mavlink import FixMavlink

if __name__ == '__main__':
    FixMavlink.extract_from_log_path(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs")