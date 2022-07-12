import os
import time
from datetime import datetime

from Cptool.config import toolConfig
from Cptool.mavlink import CollectMavlinkAPM, DroneMavlink
from Cptool.simManager import FixSimManager


def least():
    log_index = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/LASTLOG.TXT"
    with open(log_index, 'r') as f:
        i = int(f.readline())
    return i


if __name__ == '__main__':
    # Create txt if not exists
    log_index = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/LASTLOG.TXT"
    if not os.path.exists(log_index):
        with open(log_index, "w") as f:
            f.write('0')

    manager = FixSimManager(debug=toolConfig.DEBUG)

    time.sleep(1)
    while least() < 500:
        time.sleep(0.5)
        log_index = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/LASTLOG.TXT"
        if os.path.exists(log_index):
            with open(log_index, 'r') as f:
                num = int(f.readline())
        print('--------------------------------------------------------------------------------------------------')
        print(f'--------- {datetime.now()} === lastindex: {num}----------------------')
        print('--------------------------------------------------------------------------------------------------')

        manager.start_sitl()

        manager.mav_monitor_init(CollectMavlinkAPM)

        manager.mav_monitor.set_mission('Cptool/mission.txt', False)

        manager.mav_monitor.set_random_param_and_start()
        # manager.mav_monitor.start_mission()

        # manager.start_mav_monitor()

        result = manager.mav_monitor.wait_complete(remain_fail=True)

        manager.mav_monitor.reset_params()

        manager.stop_sitl()

        if not result:
            # Delete current log
            CollectMavlinkAPM.delete_current_log()
