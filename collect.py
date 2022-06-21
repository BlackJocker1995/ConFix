import logging
import os
import time
from datetime import datetime

from Cptool.config import toolConfig
from Cptool.mavlink import FixMavlink, DroneMavlink
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

        manager.mav_monitor_init(FixMavlink)

        if not manager.mav_monitor_connect():
            manager.stop_sitl()

        manager.mav_monitor.set_mission('Cptool/fitCollection.txt', False)

        manager.mav_monitor.random_param_and_set()

        manager.start_mav_monitor()

        result = manager.mav_monitor.wait_complete()
        manager.stop_sitl()

        if not result:
            # Delete current log
            DroneMavlink.delete_current_log()