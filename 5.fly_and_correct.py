import multiprocessing
import time

from Cptool.config import toolConfig
from Cptool.mavlink import FlyFixMavlinkAPM
from Cptool.simManager import FixSimManager
from ModelFit.approximate import CyTCN

if __name__ == '__main__':
    manager = FixSimManager(debug=toolConfig.DEBUG)

    # manager.start_sitl()

    mav_monitor = FlyFixMavlinkAPM(14540, multiprocessing.Queue(), multiprocessing.Queue())
    mav_monitor.connect()
    # while not mav_monitor.ready2fly():
    #     time.sleep(0.1)

    mav_monitor.init_predictor(CyTCN, 100, 128)

    mav_monitor.set_mission('Cptool/fitCollection.txt', True)

    # mav_monitor.set_random_param_and_start()
    mav_monitor.start_mission()

    mav_monitor.init_bin_log_file()

    mav_monitor.start()
    mav_monitor.online_bin_monitor()

    manager.stop_sitl()
