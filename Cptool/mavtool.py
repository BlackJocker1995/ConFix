import json
import os

import numpy as np
import pandas as pd

from Cptool.config import toolConfig


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


def get_default_values(para_dict):
    return para_dict.loc[['default']]


def select_sub_dict(para_dict, param_choice):
    return para_dict[param_choice]


def read_range_from_dict(para_dict):
    return np.array(para_dict.loc['range'].to_list())


def read_unit_from_dict(para_dict):
    return para_dict.loc['step'].to_numpy()


# Log analysis function
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


def rename_bin(log_path, ranges):
    file_list = read_path_specified_file(log_path, 'BIN')
    # 列出文件夹内所有.BIN结尾的文件并排序
    for file, num in zip(file_list, range(ranges[0], ranges[1])):
        name, _ = file.split('.')
        os.rename(f"{log_path}/{file}", f"{log_path}/{str(num).zfill(8)}.BIN")
