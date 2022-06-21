import json
import logging
import math
import multiprocessing
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import ray
from pymavlink import mavutil, mavwp
from pymavlink.DFReader import DFMessage
from pymavlink.mavutil import mavserial
from pyulog import ULog
from tqdm import tqdm

from Cptool.config import toolConfig
from ModelFit.approximate import Modeling, CyLSTM
from ModelFit.config import modelConfig


class DroneMavlink(multiprocessing.Process):
    def __init__(self, port, recv_msg_queue, send_msg_queue):
        super(DroneMavlink, self).__init__()
        self.recv_msg_queue = recv_msg_queue
        self.send_msg_queue = send_msg_queue
        self._master: mavserial = None
        self._port = port
        self.takeoff = False

    def connect(self):
        """
        Connect drone
        :return:
        """
        self._master = mavutil.mavlink_connection('udp:0.0.0.0:{}'.format(self._port))
        try:
            self._master.wait_heartbeat(timeout=30)
        except TimeoutError:
            return False
        logging.info("Heartbeat from system (system %u component %u)" % (
            self._master.target_system, self._master.target_system))
        return True

    def set_mission(self, mission_file, random: bool, timeout=30) -> bool:
        """
        Set mission
        :param random: Out of order
        :param mission_file: mission file
        :param timeout:
        :return: success
        """
        if not self._master:
            logging.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')

        loader = mavwp.MAVWPLoader()
        loader.load(mission_file)
        logging.debug(f"Load mission file {mission_file}")

        # clear the waypoint
        self._master.waypoint_clear_all_send()
        # Pop home wp if mode is PX4
        if toolConfig.MODE == 'PX4':
            loader = self.trans_wp2px4(loader)
        # send the waypoint count
        self._master.waypoint_count_send(loader.count())
        seq_list = [True] * loader.count()
        try:
            # looping to send each waypoint information
            while True in seq_list:
                msg = self._master.recv_match(type=['MISSION_REQUEST'], blocking=True,
                                              timeout=timeout)
                if msg is not None and seq_list[msg.seq] is True:
                    self._master.mav.send(loader.wp(msg.seq))
                    seq_list[msg.seq] = False
                    logging.debug(f'Sending waypoint {msg.seq}')
            mission_ack_msg = self._master.recv_match(type=['MISSION_ACK'], blocking=True, timeout=timeout)
            logging.info('Upload mission finish.')
        except TimeoutError:
            logging.warning('Upload mission timeout!')
            return False
        return True

    def start_mission(self):
        """
        Arm and start the flight
        :return:
        """
        if not self._master:
            logging.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')
        # self._master.set_mode_loiter()
        self._master.arducopter_arm()
        self._master.set_mode_auto()
        logging.info('Arm and start.')

    def set_param(self, param: str, value: float) -> None:
        """
        set a value of specific parameter
        :param param: name of the parameter
        :param value: float value want to set
        """
        if not self._master:
            raise ValueError('Connect at first!')

        self._master.param_set_send(param, value)
        self.get_param(param)

    def set_params(self, params_dict: dict) -> None:
        """
        set multiple parameter
        :param params_dict: a dict consist of {parameter:values}...
        """
        for param, value in params_dict.items():
            self.set_param(param, value)

    def get_param(self, param: str) -> float:
        """
        get current value of a parameter.
        :param param: name
        :return: value of parameter
        """
        self._master.param_fetch_one(param)
        while True:
            message = self._master.recv_match(type=['PARAM_VALUE', 'PARM'], blocking=True).to_dict()
            if message['param_id'] == param:
                logging.debug('name: %s\t value: %f' % (message['param_id'], message['param_value']))
                break
        return message['param_value']

    def get_params(self, params: list) -> dict:
        """
        get current value of a parameters.
        :param params:
        :return: value of parameter
        """
        out_dict = {}
        for param in params:
            out_dict[param] = self.get_param(param)
        return out_dict

    def get_msg(self, type, block=False):
        """
        receive the mavlink message
        :param type:
        :param block:
        :return:
        """
        msg = self._master.recv_match(type=type, blocking=block)
        return msg

    def set_mode(self, mode: str):
        """
        Set flight mode
        :param mode: string type of a mode, it will be convert to an int values.
        :return:
        """
        if not self._master:
            logging.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')
        mode_id = self._master.mode_mapping()[mode]

        self._master.mav.set_mode_send(self._master.target_system,
                                       mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                       mode_id)
        while True:
            message = self._master.recv_match(type='COMMAND_ACK', blocking=True).to_dict()
            if message['command'] == mavutil.mavlink.MAVLINK_MSG_ID_SET_MODE:
                logging.debug(f'Mode: {mode} Set successful')
                break

    def random_param_and_set(self):
        param_configuration = self.create_random_params(toolConfig.PARAM)
        self.set_params(param_configuration)
        # Unlock the uav
        self.start_mission()

    def read_status(self, status):
        out_status = dict()
        local_status = status
        while len(local_status) != 0:
            msg = self._master.recv_match(type=status, blocking=True)
            msg = msg.to_dict()
            local_status.remove(msg["mavpackettype"])
            out_status[msg["mavpackettype"]] = msg
        return out_status

    def wait_complete(self):
        pass

    @staticmethod
    def create_random_params(param_choice):
        para_dict = DroneMavlink.load_param()

        param_choice_dict = FixMavlink.select_sub_dict(para_dict, param_choice)

        out_dict = {}
        for key, param_range in param_choice_dict.items():
            value = round(random.uniform(param_range['range'][0], param_range['range'][1]) / param_range['step']) * \
                    param_range['step']
            out_dict[key] = value
        return out_dict

    @staticmethod
    def delete_current_log():
        log_index = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/LASTLOG.TXT"

        # Read last index
        with open(log_index, 'r') as f:
            num = int(f.readline())
        # To string
        num = f'{num}'
        log_file = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/{num.rjust(8, '0')}.BIN"
        # Remove file
        if os.path.exists(log_file):
            os.remove(log_file)
            # Fix last index number
            last_num = f"{int(num) - 1}"
            with open(log_index, 'w') as f:
                f.write(last_num)

    @staticmethod
    def load_param() -> json:
        """
        load parameter we want to fuzzing
        :return:
        """
        if toolConfig.MODE == 'Ardupilot':
            path = 'Cptool/param_ardu.json'
        elif toolConfig.MODE == 'PX4':
            path = 'Cptool/param_px4.json'
        with open(path, 'r') as f:
            return pd.DataFrame(json.loads(f.read()))


class FixMavlink(DroneMavlink):
    """
    Mainly responsible for initiating the communication link to interact with UAV
    """

    def __init__(self, port, recv_msg_queue, send_msg_queue):
        super(FixMavlink, self).__init__(port, recv_msg_queue, send_msg_queue)

    @staticmethod
    def log_extract_apm(msg: DFMessage):
        """
        parse the msg of mavlink
        :param msg:
        :return:
        """
        out = None
        if msg.get_type() == 'ATT':
            if len(toolConfig.LOG_MAP):
                out = {
                    'TimeS': msg.TimeUS / 1000000,
                    'Roll': math.radians(msg.Roll),
                    'Pitch': math.radians(msg.Pitch),
                    'Yaw': math.radians(msg.Yaw) ,
                }
        elif msg.get_type() == 'RATE':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                # deg to rad
                'RateRoll': math.radians(msg.R),
                'RatePitch': math.radians(msg.P),
                'RateYaw': math.radians(msg.Y),
            }
        elif msg.get_type() == 'IMU':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                'AccX': msg.AccX,
                'AccY': msg.AccY,
                'AccZ': msg.AccZ,
                'GyrX': msg.GyrX,
                'GyrY': msg.GyrY,
                'GyrZ': msg.GyrZ,
            }
        elif msg.get_type() == 'PARM':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                msg.Name: msg.Value
            }
        return out

    @staticmethod
    def extract_from_log_file(log_file):
        """
        extract log message form a bin file.
        :param log_file:
        :return:
        """
        accept_item = toolConfig.LOG_MAP

        logs = mavutil.mavlink_connection(log_file)
        # init
        out_data = []
        accpet_param = FixMavlink.load_param().columns.to_list()

        while True:
            msg = logs.recv_match(type=accept_item)
            if msg is None:
                break
            if msg.get_type() == 'ATT':
                out_data.append(FixMavlink.log_extract_apm(msg))
            elif msg.get_type() == 'RATE':
                out_data.append(FixMavlink.log_extract_apm(msg))
            elif msg.get_type() == 'IMU':  # and msg['I'] == 0:
                out_data.append(FixMavlink.log_extract_apm(msg))
            elif msg.get_type() == 'PARM' and msg.Name in accpet_param:
                out_data.append(FixMavlink.log_extract_apm(msg))

        pd_array = pd.DataFrame(out_data)

        # Remain timestamp .1 and drop duplicate
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        pd_array = pd_array.drop_duplicates(keep='first')

        # merge data in same TimeS
        df_array = pd.DataFrame(columns=pd_array.columns)
        for group, group_item in pd_array.groupby('TimeS'):
            # fillna
            group_item = group_item.fillna(method='ffill')
            group_item = group_item.fillna(method='bfill')
            df_array = df_array.append(group_item.mean(), ignore_index=True)
        # Drop nan
        df_array = df_array.fillna(method='ffill')
        df_array = df_array.dropna()

        # Sort name
        attitude_name = ['TimeS', 'Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw',
                         'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ'
                         ]
        param_seq = FixMavlink.load_param().columns.to_list()
        param_name = df_array.keys().difference(attitude_name).to_list()
        param_name.sort(key=lambda item: param_seq.index(item))
        # Status value + Parameter name
        attitude_name.extend(param_name)
        # Switch sequence and return
        return df_array[attitude_name]

    @staticmethod
    def read_path_specified_file(log_path, exe):
        """
        :param log_path:
        :param exe:
        :return:
        """
        file_list = []
        for filename in os.listdir(log_path):
            if filename.endswith(f'.{exe}'):
                file_list.append(filename)
        file_list.sort()
        return file_list

    @staticmethod
    def extract_from_log_path(log_path, threat=None):
        """
        extract and convert bin file to csv
        :param log_path:
        :param threat: multiple threat
        :return:
        """

        file_list = FixMavlink.read_path_specified_file(log_path, 'BIN')
        if not os.path.exists(f"{log_path}/csv"):
            os.makedirs(f"{log_path}/csv")

        # 列出文件夹内所有.BIN结尾的文件并排序
        for file in tqdm(file_list):
            name, _ = file.split('.')
            if os.path.exists(f'{log_path}/csv/{name}.csv'):
                continue
            # extract
            try:
                csv_data = FixMavlink.extract_from_log_file(log_path + f'/{file}')
                csv_data.to_csv(f'{log_path}/csv/{name}.csv', index=False)
            except Exception as e:
                logging.warning(f"Error processing {file} : {e}")
                continue

    @staticmethod
    @ray.remote
    def extract_from_log_path_threat(log_path, file_list):
        for file in file_list:
            name, _ = file.split('.')
            # if os.path.exists(f'{log_path}/csv/{name}.csv'):
            #     continue
            csv_data = FixMavlink.extract_from_log_file(log_path + f'/{file}')
            # 只保留一位小数
            # csv_data = csv_data[['TimeS', 'Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw',
            #                      'DesRoll', 'DesPitch', 'DesYaw', 'DesRateRoll', 'DesRatePitch', 'DesRateYaw']]
            csv_data.to_csv(f'{log_path}/csv/{name}.csv', index=False)
            print(f"\r{log_path} Process: {name}")
        if os.path.exists(f'{log_path}/mark.pkl'):
            shutil.copyfile(f'{log_path}/mark.pkl', f'{log_path}/csv/mark.pkl')
        return True

    @staticmethod
    def extract_from_ulog(log_file):
        """
        extract and convert ulog file to csv
        :param log_path:
        :return:
        """
        # load ulog
        ulog = ULog(log_file)
        att = pd.DataFrame(ulog.get_dataset('vehicle_attitude_setpoint').data)
        rate = pd.DataFrame(ulog.get_dataset('vehicle_rates_setpoint').data)
        acc = pd.DataFrame(ulog.get_dataset('vehicle_acceleration').data)

        # 给标记
        att = att[['timestamp', 'roll_body', 'pitch_body', 'yaw_body']]
        att['label'] = np.zeros(len(att))
        rate = rate[['timestamp', 'roll', 'pitch', 'yaw']]
        rate['label'] = np.zeros(len(rate)) + 1
        acc = acc[['timestamp', 'xyz[0]', 'xyz[1]', 'xyz[2]']]
        acc['label'] = np.zeros(len(acc)) + 2

        # 合并到一个表中
        array = att.append(rate, ignore_index=True)
        array = array.append(acc, ignore_index=True)
        array = array.sort_values(by='timestamp').reset_index(drop=True)

        # 找出重复的index
        pre = array['label'].to_numpy()[:-1]
        next = array['label'].to_numpy()[1:]
        # 去重
        array = array.iloc[:-1][(pre - next) != 0]
        label = array['label'].to_numpy()

        data = []
        for i in range(len(label) - 2):
            if label[i:i + 3].sum() == 3:
                out = FixMavlink.log_extract_px4(array.iloc[i:i + 3])
                data.append(out)
        data = pd.DataFrame(data, columns=['timestamp', 'xyz[0]', 'xyz[1]', 'xyz[2]',
                                           'roll_body', 'pitch_body', 'pitch_body',
                                           'roll', 'pitch', 'yaw'])
        data.rename(columns={
            'timestamp': 'TimeS',
            'xyz[0]': 'AccX',
            'xyz[1]': 'AccY',
            'xyz[2]': 'AccZ',
            'roll_body': 'Roll',
            'pitch_body': 'Pitch',
            'yaw_body': 'Yaw',
            'roll': 'RateRoll',
            'pitch': 'RatePitch',
            'yaw': 'RateYaw',
        }, inplace=True)
        return data

    @staticmethod
    def random_param_value(param_json: dict):
        """
        random create the value
        :param param_json:
        :return:
        """
        out = {}
        for name, item in param_json.items():
            range = item['range']
            step = item['step']
            random_sample = random.randrange(range[0], range[1], step)
            out[name] = random_sample
        return out

    @staticmethod
    def get_default_values(para_dict):
        return para_dict.loc[['default']]

    @staticmethod
    def select_sub_dict(para_dict, param_choice):
        return para_dict[param_choice]

    @staticmethod
    def read_range_from_dict(para_dict):
        return np.array(para_dict.loc['range'].to_list())

    @staticmethod
    def read_unit_from_dict(para_dict):
        return para_dict.loc['step'].to_numpy()

    def wait_complete(self):
        if not self._master:
            raise ValueError('Connect at first!')
        while True:
            try:
                message = self._master.recv_match(type=['STATUSTEXT'],
                                                  blocking=True, timeout=30)
                if message is not None:
                    message = message.to_dict()
                    out_msg = "None"
                    line = message['text']
                    if message["severity"] == 6:
                        if "Land" in line:
                            # if successful landed, break the loop and return true
                            logging.info(f"Successful.")
                            return True
                    elif message["severity"] == 2 or message["severity"] == 0:
                        # Appear error, break loop and return false
                        if "SIM Hit ground at" in line:
                            pass
                        elif "Potential Thrust Loss" in line:
                            pass
                        elif "PreArm" in line:
                            pass
                            # will not generate log file
                            logging.info(f"Get error with {message['text']}")
                            return True
                        logging.info(f"Get error with {message['text']}")
                        return False
            except TimeoutError:
                # Mission point time out, change other params
                logging.warning('wp timeout!')
                return False
            except KeyboardInterrupt:
                logging.info('Key bordInterrupt! exit')
                return False
        return True


class FlyFixMavlink(DroneMavlink):
    def __init__(self, port, recv_msg_queue, send_msg_queue):
        super(FlyFixMavlink, self).__init__(port, recv_msg_queue, send_msg_queue)
        self.predictor: CyLSTM = None

    def init_predictor(self, epochs, batch_size):
        self.predictor = CyLSTM(epochs, batch_size, toolConfig.DEBUG)
        self.predictor.read_model()

    def read_status_patch(self, time_unit, status):
        out_data = []
        first_msg = self._master.recv_match(type=status, blocking=True)
        out_data.append(FlyFixMavlink.runtime_extract_apm(first_msg))
        first_time = FlyFixMavlink.get_time_index(first_msg)
        new_time = first_time

        # Collect data in one time_unit
        while new_time <= first_time + time_unit:
            new_msg = self._master.recv_match(type=status, blocking=True)
            # Add and process
            new_time = FlyFixMavlink.get_time_index(new_msg)
            out_data.append(FlyFixMavlink.runtime_extract_apm(new_msg))

        # Read current configuration
        params = self.get_params(toolConfig.PARAM)
        out_data.append(params)
        pd_array = pd.DataFrame(out_data)

        # Remain timestamp .1 and drop duplicate
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        pd_array = pd_array.drop_duplicates(keep='first')
        pd_array[toolConfig.PARAM] = pd_array[toolConfig.PARAM].fillna(method="bfill")

        # merge data in same TimeS
        df_array = pd.DataFrame(columns=pd_array.columns)
        for group, group_item in pd_array.groupby('TimeS'):
            # fillna
            group_item = group_item.fillna(method='ffill')
            group_item = group_item.fillna(method='bfill')
            df_array = df_array.append(group_item.mean(), ignore_index=True)
        # Drop nan
        df_array = df_array.fillna(method='ffill')
        df_array = df_array.dropna()

        return df_array

    def cal_patch_loss(self, status_data, predicted_data) -> float:
        # TODO
        return 0

    @staticmethod
    def runtime_extract_apm(msg):
        """
        parse the msg of mavlink
        :param msg:
        :return:
        """
        out = None
        if msg.name == 'ATTITUDE':
            if len(toolConfig.LOG_MAP):
                out = {
                    'TimeS': msg.time_boot_ms / 1000,
                    # Rad
                    'Roll': msg.roll,
                    'Pitch': msg.pitch,
                    'Yaw': msg.pitch,
                    'RateRoll': msg.rollspeed,
                    'RatePitch': msg.pitchspeed,
                    'RateYaw': msg.yawspeed,
                }
        elif msg.name == 'RAW_IMU':
            out = {
                'TimeS': msg.time_usec / 1000000,
                # raw
                'AccX': msg.xacc,
                'AccY': msg.yacc,
                'AccZ': msg.zacc,
                'GyrX': msg.xgyro,
                'GyrY': msg.ygyro,
                'GyrZ': msg.zgyro,
            }
        return out

    def predict_status(self, status_data):
        if self.predictor is None:
            logging.warning('Predictor is not set!')
            raise ValueError('Train or load model at first')

        status_numpy = status_data.to_numpy()
        predict_X = self.predictor.predict(status_numpy)

        # data retrans
        if modelConfig.RETRANS:
            trans = self.predictor.load_trans()
            predict_X = trans.inverse_transform(predict_X)
        return predict_X

    def sample_state(self):
        self.read_status(["ATTITUDE", "NAV_CONTROLLER_OUTPUT", "VFR_HUD"])

    def detect_instability(self, status_data):
        predicted_data = self.predict_status(status_data)
        # 计算偏差
        patch_loss = self.cal_patch_loss(status_data, predicted_data)
        # If over the threshold
        pass


    def repair_configuration(self):
        # TODO
        pass

    @staticmethod
    def get_time_index(msg):
        """
        As different message have different time unit. It needs to convert to same second unit.
        :return:
        """
        if msg.name == "ATTITUDE":
            return msg.time_boot_ms / 1000
        if msg.name == "RAW_IMU":
            return msg.time_usec / 1000000

    def wait_complete(self):
        if not self._master:
            raise ValueError('Connect at first!')
        while True:
            try:
                message = self._master.recv_match(type=['STATUSTEXT'],
                                                  blocking=True, timeout=30)
                if message is not None:
                    message = message.to_dict()
                    out_msg = "None"
                    line = message['text']
                    if message["severity"] == 6:
                        if "Land" in line:
                            # if successful landed, break the loop and return true
                            logging.info(f"Successful.")
                            return True
                    # elif message["severity"] == 2 or message["severity"] == 0:
                    #     # Appear error, break loop and return false
                    #     if "SIM Hit ground at" in line:
                    #         pass
                    #     elif "Potential Thrust Loss" in line:
                    #         pass
                    #     elif "PreArm" in line:
                    #         pass
                    #         # will not generate log file
                    #         logging.info(f"Get error with {message['text']}")
                    #         return True
                    #     logging.info(f"Get error with {message['text']}")
                    #     return False
            except TimeoutError:
                # Mission point time out, change other params
                logging.warning('wp timeout!')
                return False
            except KeyboardInterrupt:
                logging.info('Key bordInterrupt! exit')
                return False
        return True

    def run(self) -> None:
        # loop to change configuration

        # Sample a patch
        status_data = self.read_status_patch(2, ["ATTITUDE", "RAW_IMU"])
        # Detect
        result = self.detect_instability(status_data)
        # # GA Repair
        # if result is True:
        #     self.repair_configuration()
