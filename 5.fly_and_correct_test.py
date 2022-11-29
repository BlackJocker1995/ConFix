import multiprocessing
import time

from Cptool.config import toolConfig
from Cptool.mavlink import FlyFixMavlinkAPM
from Cptool.simManager import FixSimManager
from ModelFit.approximate import CyTCN

if __name__ == '__main__':
    manager = FixSimManager(debug=toolConfig.DEBUG)

    manager.start_sitl()

    mav_monitor = FlyFixMavlinkAPM(14540, multiprocessing.Queue(), multiprocessing.Queue())
    mav_monitor.connect()
    while not mav_monitor.ready2fly():
        time.sleep(0.1)

    mav_monitor.init_predictor(CyTCN, 100, 128)

    mav_monitor.set_mission('Cptool/fitCollection.txt', False)

    v = {
        "PSC_VELXY_P": 6.0,
        "PSC_VELXY_I": 0.1,
        "PSC_VELXY_D": 0.394,
        "PSC_ACCZ_P": 0.122,
        "PSC_ACCZ_I": 0.357,
        "ATC_ANG_RLL_P": 7.9,
        "ATC_RAT_RLL_P": 0.495,
        "ATC_RAT_RLL_I": 1.61,
        "ATC_RAT_RLL_D": 0.036,
        "ATC_ANG_PIT_P": 8.0,
        "ATC_RAT_PIT_P": 0.415,
        "ATC_RAT_PIT_I": 0.99,
        "ATC_RAT_PIT_D": 0.05,
        "ATC_ANG_YAW_P": 2.9,
        "ATC_RAT_YAW_P": 2.495,
        "ATC_RAT_YAW_I":0.01,
        "ATC_RAT_YAW_D": 0.009,
        "WPNAV_SPEED":  800,
        "WPNAV_ACCEL": 500,
        "ANGLE_MAX": 5250
    }

    mav_monitor.set_params(v)

    mav_monitor.start_mission()

    mav_monitor.init_binary_log_file()

    # mav_monitor.start()
    mav_monitor.online_bin_monitor()

    manager.stop_sitl()
