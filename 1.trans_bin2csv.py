import os

import numpy as np
import pandas as pd

from Cptool.boardMavlink import BoardMavlinkAPM
from Cptool.config import toolConfig

if __name__ == '__main__':
    BoardMavlinkAPM.extract_log_path(f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_train", skip=True,
                                     keep_des=False, thread=6)
