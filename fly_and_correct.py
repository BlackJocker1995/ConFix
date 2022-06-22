import logging
import os
import time
from datetime import datetime

import ray

from Cptool.config import toolConfig
from Cptool.mavlink import FixMavlink, DroneMavlink, FlyFixMavlink
from Cptool.simManager import FixSimManager


if __name__ == '__main__':
    manager = FixSimManager(debug=toolConfig.DEBUG)

    # manager.start_sitl()

    manager.mav_monitor_init(FlyFixMavlink)

    manager.mav_monitor.init_predictor(100, 128)

    manager.mav_monitor.set_mission('Cptool/fitCollection.txt', False)

    manager.start_mav_monitor()

    manager.mav_monitor.online_monitor()
