"""
SimManager Version: 1.2
"""
import logging
import math
import multiprocessing
import os
import re
import sys
import threading
import time

import eventlet
import pexpect
from pexpect import spawn
from pymavlink import mavextra, mavwp

from Cptool.config import toolConfig
from Cptool.mavlink import FixMavlink


class SimSimulator(multiprocessing.Process):
    def __init__(self, recv_msg_queue: multiprocessing.Queue, send_msg_queue: multiprocessing.Queue):
        super(SimSimulator, self).__init__()
        self.recv_msg_queue = recv_msg_queue
        self.send_msg_queue = send_msg_queue

        if toolConfig.DEBUG:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO)


class GaSimSimulator(SimSimulator):
    def __init__(self, recv_msg_queue: multiprocessing.Queue, send_msg_queue: multiprocessing.Queue):
        super(SimSimulator, self).__init__(recv_msg_queue, send_msg_queue)

