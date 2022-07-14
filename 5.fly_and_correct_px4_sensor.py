import multiprocessing
import time

from Cptool.config import toolConfig
from Cptool.mavlink import FlyFixMavlinkAPM, FlyFixMavlinkPX4
from Cptool.simManager import FixSimManager
from ModelFit.approximate import CyTCN

if __name__ == '__main__':
    toolConfig.select_mode("PX4")
    manager = FixSimManager(debug=toolConfig.DEBUG)

    manager.start_sitl()

    manager.mav_monitor_init(FlyFixMavlinkPX4)

    manager.mav_monitor.init_predictor(CyTCN, 100, 128)

    manager.mav_monitor.set_mission('Cptool/mission_px4.txt', False)

    manager.mav_monitor.start_mission()

    manager.mav_monitor.init_bin_log_file()

    # mav_monitor.start()
    manager.mav_monitor.online_ulg_monitor()

    manager.stop_sitl()
