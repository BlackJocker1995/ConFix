import argparse
import logging
import os
import signal
import time
from datetime import datetime

from Cptool.config import toolConfig
from Cptool.mavlink import MavlinkAPM, MavlinkPX4
from Cptool.simManager import FixSimManager
import psutil

def try_kill_mavproxy(device):
    device = int(device)
    for proc in psutil.process_iter():
        try:
            cmdline = proc.cmdline()
            pid = proc.pid
        except psutil.NoSuchProcess:
            continue

        if (len(cmdline) >= 2 and
            os.path.basename(cmdline[1]) == "mavproxy.py") and \
                str(14540 + device) in cmdline[3]:
            os.kill(pid, signal.SIGKILL)

if __name__ == '__main__':
    toolConfig.select_mode("PX4")
    # Create txt if not exists

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--device', dest='device', type=str, help='Name of the candidate')
    args = parser.parse_args()
    device = args.device
    if device is None:
        device = 0

    # Manager
    manager = FixSimManager(debug=toolConfig.DEBUG)
    for i in range(500):
        print('--------------------------------------------------------------------------------------------------')
        print(f'--------- {datetime.now()} ===----------------------')
        print('--------------------------------------------------------------------------------------------------')

        # init environment
        manager.start_multiple_sitl(device)
        manager.start_multiple_sim(device)
        manager.online_mavlink_init(MavlinkPX4, device)
        manager.mav_monitor_init(int(14030) + int(device))
        manager.board_mavlink_init()

        if toolConfig.HOME is None:
            mission_file = 'Cptool/mission_px4.txt'
        else:
            mission_file = 'Cptool/fitCollection_px4.txt'
        # Set mission_ file
        set_result = manager.online_mavlink.set_mission(mission_file, False)

        if not set_result:
            logging.warning("Mission file set failed!")
            continue

        time.sleep(2)
        manager.online_mavlink.set_random_param_and_start()
        # manager.online_mavlink.start_mission()
        #
        result = manager.mav_monitor.run()

        manager.online_mavlink.reset_params()

        try_kill_mavproxy(device)
        manager.stop_sitl()
        manager.stop_sim()
        if not result:
            # Delete current log
            manager.board_mavlink.delete_current_log(device)
        # keep unstable only
        # if not result:
        #     # Delete current log
        #     manager.board_mavlink.delete_current_log(device)
