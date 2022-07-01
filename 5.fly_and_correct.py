import logging
import os
import time
from datetime import datetime

import multiprocessing
import ray

from Cptool.config import toolConfig
from Cptool.mavlink import FixMavlink, DroneMavlink, FlyFixMavlink
from Cptool.simManager import FixSimManager


if __name__ == '__main__':
    manager = FixSimManager(debug=toolConfig.DEBUG)

    manager.start_sitl()

    mav_monitor = FlyFixMavlink(14540, multiprocessing.Queue(), multiprocessing.Queue())
    mav_monitor.connect()
    while not mav_monitor.ready2fly():
        time.sleep(0.1)

    mav_monitor.init_predictor(100, 128)

    mav_monitor.set_mission('Cptool/mission.txt', True)

    mav_monitor.start_mission()

    mav_monitor.init_bin_log_file()

    # mav_monitor.start()
    mav_monitor.online_bin_monitor()

    manager.stop_sitl()
