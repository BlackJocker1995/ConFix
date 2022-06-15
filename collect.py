import logging
import time
from datetime import datetime

from Cptool.config import toolConfig
from Cptool.mavlink import FixMavlink, DroneMavlink
from Cptool.simManager import FixSimManager

if __name__ == '__main__':
    manager = FixSimManager(debug=toolConfig.DEBUG)

    time.sleep(1)
    i = 0
    while True:
        time.sleep(1)
        print('--------------------------------------------------------------------------------------------------')
        print(f'--------- {datetime.now()} === {i}----------------------')
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
        i += 1
