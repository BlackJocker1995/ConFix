"""
SimManager Version: 4.0
"""
import logging
import multiprocessing
import os
import time

import pexpect
from pexpect import spawn

from Cptool.config import toolConfig
from Cptool.mavlink import DroneMavlink
from Cptool.simSimulator import SimSimulator


class SimManager:
    def __init__(self, debug: bool = False):
        self._sim_task = None
        self._sitl_task = None
        self.sim_monitor: SimSimulator = None
        self.mav_monitor: DroneMavlink = None
        self._even = None
        self.sim_msg_queue = multiprocessing.Queue()
        self.mav_msg_queue = multiprocessing.Queue()

        if debug:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.INFO)

    def start_sim(self):
        """
        启动AIRSIM_PATH目录下的Airsim模拟器
        :return:
        """
        # Airsim
        cmd = None
        if toolConfig.SIM == 'Airsim':
            cmd = f'gnome-terminal -- {toolConfig.AIRSIM_PATH} ' \
                  f'-ResX={toolConfig.HEIGHT} -ResY={toolConfig.WEIGHT} -windowed'
        if toolConfig.SIM == 'Jmavsim':
            cmd = f'gnome-terminal -- bash /home/rain/PX4-Autopilot/Tools/jmavsim_run.sh'
        if toolConfig.SIM == 'Morse':
            cmd = f'gnome-terminal -- morse run /home/rain/ardupilot/libraries/SITL/examples/Morse/quadcopter.py'
        if toolConfig.SIM == 'Gazebo':
            cmd = f'gnome-terminal -- gazebo --verbose worlds/iris_arducopter_runway.world'
        if cmd is None:
            raise ValueError('Not support mode')
        logging.info(f'Start Simulator {toolConfig.SIM}')
        self._sim_task = pexpect.spawn(cmd, cwd='/home/rain/')

    def start_sitl(self):
        """
        启动软件在环 模拟器，分为PX4 和 Ardupilot
        :return:
        """
        if os.path.exists(f"{toolConfig.ARDUPILOT_LOG_PATH}/eeprom.bin"):
            os.remove(f"{toolConfig.ARDUPILOT_LOG_PATH}/eeprom.bin")
        if os.path.exists(f"{toolConfig.ARDUPILOT_LOG_PATH}/mav.parm"):
            os.remove(f"{toolConfig.ARDUPILOT_LOG_PATH}/mav.parm")

        cmd = None
        if toolConfig.MODE == 'Ardupilot':
            if toolConfig.SIM == 'Airsim':
                if toolConfig.HOME is not None:
                    cmd = f"python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py -v ArduCopter " \
                          f"--location={toolConfig.HOME}" \
                          f" -f airsim-copter --out=127.0.0.1:14550 --out=127.0.0.1:14540 " \
                          f" -S {toolConfig.SPEED}"
                else:
                    cmd = f"python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py -v ArduCopter -f airsim-copter " \
                          f"--out=127.0.0.1:14550 --out=127.0.0.1:14540 -S {toolConfig.SPEED}"
            if toolConfig.SIM == 'Morse':
                cmd = f"python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py -v ArduCopter --model morse-quad " \
                      f"--add-param-file=/home/rain/ardupilot/libraries/SITL/examples/Morse/quadcopter.parm  " \
                      f"--out=127.0.0.1:14550 -S {toolConfig.SPEED}"
            if toolConfig.SIM == 'Gazebo':
                cmd = f'python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py -f gazebo-iris -v ArduCopter ' \
                      f'--out=127.0.0.1:14550 -S {toolConfig.SPEED}'
            if toolConfig.SIM == 'SITL':
                if toolConfig.HOME is not None:
                    cmd = f"python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py --location={toolConfig.HOME} " \
                          f"--out=127.0.0.1:14550 --out=127.0.0.1:14540 -v ArduCopter -w -S {toolConfig.SPEED} "
                else:
                    cmd = f"python3 /home/rain/ardupilot/Tools/autotest/sim_vehicle.py " \
                          f"--out=127.0.0.1:14550 --out=127.0.0.1:14540 -v ArduCopter -w -S {toolConfig.SPEED} "
            self._sitl_task = pexpect.spawn(cmd, cwd=toolConfig.ARDUPILOT_LOG_PATH, timeout=30, encoding='utf-8')

        if toolConfig.MODE == 'PX4':
            pre_argv = f"PX4_HOME_LAT=-35.362758 "\
                        f"PX4_HOME_LON=149.165135 "\
                        f"PX4_HOME_ALT=583.730592 "\
                        f"PX4_SIM_SPEED_FACTOR={toolConfig.SPEED}"
            # for arg in pre_argv:
            #     os.system(arg)
            #     time.sleep(0.3)
            if toolConfig.SIM == 'Airsim':
                cmd = f'make px4_sitl_default none_iris'
            if toolConfig.SIM == 'Jmavsim':
                cmd = f'make {pre_argv} px4_sitl_default jmavsim'

            self._sitl_task = pexpect.spawn(cmd, cwd=toolConfig.PX4_RUN_PATH, timeout=30, encoding='utf-8')
        logging.info(f"Start {toolConfig.MODE} --> [{toolConfig.SIM}]")
        if cmd is None:
            raise ValueError('Not support mode or simulator')

    def sim_monitor_init(self, simulator_class):
        """
        初始化airsim监控器
        :return:
        """
        self.sim_monitor = simulator_class(recv_msg_queue=self.mav_msg_queue, send_msg_queue=self.sim_msg_queue)
        time.sleep(3)

    def mav_monitor_init(self, mavlink_class: DroneMavlink = DroneMavlink):
        """
        初始化SITL在环
        :return:
        """
        self.mav_monitor = mavlink_class(14540, recv_msg_queue=self.sim_msg_queue, send_msg_queue=self.mav_msg_queue)
        self.mav_monitor.connect()
        if toolConfig.MODE == 'Ardupilot':
            if self.mav_monitor.ready2fly():
                return True
        elif toolConfig.MODE == 'PX4':
            while True:
                line = self._sitl_task.readline()
                if 'notify' in line:
                    # Disable the fail warning and return
                    self._sitl_task.send("param set NAV_RCL_ACT 0 \n")
                    time.sleep(0.1)
                    self._sitl_task.send("param set NAV_DLL_ACT 0 \n")
                    return True

    def mav_monitor_connect(self):
        """
        Mavlnik连接
        :return:
        """
        return self.mav_monitor.connect()

    def mav_monitor_start_mission(self):
        """
        开始任务
        :return:
        """
        self.mav_monitor.start_mission()

    def sim_monitor_confirm_api(self):
        self.sim_monitor.confirm_api()

    def sim_monitor_reset_item(self):
        self.sim_monitor.reset_item()

    def sim_close_msg(self):
        pass

    def change_sitl_wind(self, direction=60, speed=10):
        self._sitl_task.send(f'param set SIM_WIND_DIR {direction} \n')
        time.sleep(0.1)
        self._sitl_task.send(f'param set SIM_WIND_SPD {speed} \n')
        logging.info(f"Wind change to direction {direction} and {speed} m/s")

    def stop_sitl(self):
        self._sitl_task.sendcontrol('c')
        while True:
            line = self._sitl_task.readline()
            if not line:
                break
        self._sitl_task.close(force=True)
        logging.info('Stop SITL task.')
        self.sim_close_msg()
        logging.debug('Send mavclosed to Airsim.')

    def sitl_task(self) -> spawn:
        return self._sitl_task

    def airsim_task(self) -> spawn:
        return self._sim_task


class FixSimManager(SimManager, multiprocessing.Process):

    def __init__(self, debug: bool = False):
        super(FixSimManager, self).__init__(debug)
        super(multiprocessing.Process, self).__init__()

    def mav_monitor_set_mission(self, mission_file, random: bool = False):
        """
        设置任务
        :param mission_file:任务路径
        :param random:任务是否乱序
        :return:
        """
        return self.mav_monitor.set_mission(mission_file, random)

    def mav_monitor_set_random_param(self):
        """
        初始化airsim监控器
        :return:
        """
        params_dict = self.mav_monitor.load_param()
        params_value = self.mav_monitor.random_param_value(params_dict)
        self.mav_monitor.set_params(params_value)

    def sim_monitor_set_wind(self, button, top):
        self.sim_monitor.set_wind_random(button, top)

    def start_sim_monitor(self):
        """
        启动Airsim监控进程
        :return:
        """
        self.sim_monitor.start()

    def start_mav_monitor(self):
        """
        启动mavlink监控进程
        :return:
        """
        self.mav_monitor.start()

    def one_step_mav_monitor(self):
        """
        一键启动
        :return:
        """
        self.start_sitl()
        self.mav_monitor_init()
        self.mav_monitor_connect()

    def one_step_sim_monitor(self):
        """
        一键启动
        :return:
        """
        self.start_sim()
        self.sim_monitor_init()
        self.sim_monitor_confirm_api()

    def run(self) -> None:
        while True:
            line = self._sitl_task.readline()
            time.sleep(0.1)
            print(line)
