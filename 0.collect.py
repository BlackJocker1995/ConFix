import argparse
import logging
import os
from datetime import datetime

from Cptool.config import toolConfig
from Cptool.mavlink import MavlinkAPM
from Cptool.simManager import FixSimManager

# Create txt if not exists
def least():
    log_index = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/LASTLOG.TXT"
    with open(log_index, 'r') as f:
        i = int(f.readline())
    return i

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--device', dest='device', type=str, help='Name of the candidate')
    args = parser.parse_args()
    device = args.device
    if device is None:
        device = 0

    # Manager
    manager = FixSimManager(debug=toolConfig.DEBUG)
    while least() < 500:
        log_index = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/LASTLOG.TXT"
        if os.path.exists(log_index):
            with open(log_index, 'r') as f:
                num = int(f.readline())
        print('--------------------------------------------------------------------------------------------------')
        print(f'--------- {datetime.now()} === lastindex: {num}----------------------')
        print('--------------------------------------------------------------------------------------------------')

        # init environment
        manager.start_multiple_sitl(device)

        manager.online_mavlink_init(MavlinkAPM, device)
        manager.mav_monitor_init(int(14560) + int(device))
        manager.board_mavlink_init()

        if toolConfig.HOME is None:
            mission_file = 'Cptool/mission.txt'
        else:
            mission_file = 'Cptool/fitCollection.txt'
        # Set mission_ file
        set_result = manager.online_mavlink.set_mission(mission_file, False)

        if not set_result:
            logging.warning("Mission file set failed!")
            continue

        manager.online_mavlink.set_random_param_and_start()

        # monitor error
        # manager.mav_monitor.start()

        result = manager.mav_monitor.run()

        manager.online_mavlink.reset_params()

        manager.stop_sitl()

        if not result:
            # Delete current log
            manager.board_mavlink.delete_current_log(device)
