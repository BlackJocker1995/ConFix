import multiprocessing
import time

from Cptool.config import toolConfig
from Cptool.mavlink import FlyFixMavlinkAPM
from Cptool.simManager import FixSimManager
from ModelFit.approximate import CyTCN

if __name__ == '__main__':
    manager = FixSimManager(debug=toolConfig.DEBUG)

    manager.start_sitl()

    manager.mav_monitor_init(FlyFixMavlinkAPM)

    manager.mav_monitor.init_predictor(CyTCN, 100, 128)

    manager.mav_monitor.set_mission('Cptool/fitCollection.txt', False)

    manager.mav_monitor.start_mission()

    # manager.mav_monitor.init_bin_log_file()

    time.sleep(6)
    manager.change_sitl_wind(speed=15)
    # mav_monitor.start()
    # manager.mav_monitor.online_bin_monitor()

    manager.stop_sitl()
