import json

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