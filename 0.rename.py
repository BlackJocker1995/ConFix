from Cptool.config import toolConfig
from Cptool.mavtool import rename_bin

if __name__ == '__main__':
    rename_bin(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_regular&unstable", [501, 1000])
