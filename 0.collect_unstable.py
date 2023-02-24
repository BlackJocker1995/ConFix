import argparse
import csv
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd

import Cptool
import ModelFit
from Cptool.config import toolConfig
from sklearn.utils import shuffle

from Cptool.mavlink import FlyFixMavlinkAPM, CollectMavlinkAPM
from Cptool.simManager import SimManager, FixSimManager
from ModelFit.approximate import CyTCN

if __name__ == '__main__':
    # Device
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--device', dest='device', type=str, help='Name of the candidate')
    args = parser.parse_args()
    device = args.device
    if device is None:
        device = 0
    # Read incorrect configuration
    configurations = pd.read_csv(f"validation/{toolConfig.MODE}/params.csv")
    incorrect_configuration = configurations[configurations["result"] != "pass"]

    # Stochastic order
    incorrect_configuration = shuffle(incorrect_configuration)

    for index, row in incorrect_configuration.iterrows():
        # Simulation Manager
        manager = FixSimManager(debug=toolConfig.DEBUG)

        time.sleep(1)
        print(f'======================={index} / {incorrect_configuration.shape[0]} ==========================')
        config = row.drop(["score", "result"]).astype(float)

        if os.path.exists(f'validation/{toolConfig.MODE}/finish.csv'):
            while not os.access(f"validation/{toolConfig.MODE}/finish.csv", os.R_OK):
                continue
            exit_data = pd.read_csv(f'validation/{toolConfig.MODE}/finish.csv')
            # If the value has been validate
            if ((exit_data - config).sum(axis=1).abs() < 0.00001).sum() > 0:
                continue
            if exit_data.shape[0] > 499:
                break

        config = config.to_dict()
        # start multiple SITL
        manager.start_multiple_sitl(device)
        manager.mav_monitor_init(CollectMavlinkAPM, device)

        manager.mav_monitor.set_mission('Cptool/fitCollection.txt', False)

        manager.mav_monitor.set_params(config)

        manager.mav_monitor.start_mission()

        result = manager.mav_monitor.wait_complete()

        manager.mav_monitor.reset_params()

        manager.stop_sitl()

        # if the result have no instability, skip.
        if not os.path.exists(f'validation/{toolConfig.MODE}/finish.csv'):
            data = pd.DataFrame(columns=toolConfig.PARAM)
            data.to_csv(f'validation/{toolConfig.MODE}/finish.csv', index=False)

        while not os.access(f"validation/{toolConfig.MODE}/finish.csv", os.W_OK):
            continue
        # Add instability result
        tmp_row = list(config.values())

        # Write Row
        with open(f"validation/{toolConfig.MODE}/finish.csv", 'a+') as f:
            csv_file = csv.writer(f)
            csv_file.writerow(tmp_row)
            logging.debug("Write row to params.csv.")
