import multiprocessing
import time

from Cptool.config import toolConfig
from Cptool.mavlink import FlyFixMavlink
from Cptool.simManager import FixSimManager
from Cptool.simSimulator import GaSimSimulator
from ModelFit.approximate import CyTCN

if __name__ == '__main__':
    manager = FixSimManager(debug=toolConfig.DEBUG)

    manager.start_sim()
    manager.sim_monitor_init(GaSimSimulator)

    manager.start_sitl()

    manager.mav_monitor_init(FlyFixMavlink)
    manager.sim_monitor_confirm_api()

    manager.mav_monitor_set_mission("Cptool/mission.txt", random=False)

    # manager.start_sim_monitor()
    # manager.start_mav_monitor()

    manager.mav_monitor.init_predictor(CyTCN, 100, 128)

    manager.mav_monitor.init_bin_log_file()

    manager.mav_monitor_start_mission()

    time.sleep(10)

    manager.sim_monitor.start()

    manager.mav_monitor.online_bin_monitor()

    while manager.mav_monitor.is_alive():
        time.sleep(0.1)
    manager.stop_sitl()
