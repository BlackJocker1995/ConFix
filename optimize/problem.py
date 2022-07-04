import logging
import geatpy as ea
import numpy as np
import pandas as pd

from Cptool.config import toolConfig
from Cptool.mavtool import load_param, select_sub_dict
from ModelFit.approximate import CyLSTM


class Problem():
    def __init__(self):
        self.status_data: pd.DataFrame = None
        self.predictor: CyLSTM = None
        self.param_bounds = None
        self.step = None

    def init_status(self, status_data):
        self.status_data = status_data

    def init_predictor(self, predictor):
        self.predictor = predictor
        self.predictor.read_trans()

    def init_bounds_and_step(self, param_bounds, step):
        self.param_bounds = param_bounds
        # step
        self.step = step

    def param_value2step(self, configuration):
        np_config = np.ceil(configuration / self.step)
        np_config = np_config * self.step
        np_config = pd.DataFrame([np_config.tolist()], columns=toolConfig.PARAM)
        return np_config.iloc[0].to_dict()

    def function(self, configuration):
        pass


class DTWLossProblem(Problem):
    def __init__(self):
        super().__init__()

    def function(self, configuration):
        configuration = self.param_value2step(configuration)
        logging.debug(f"Optimizer configuration: {configuration}")
        # replace parameter value
        try_statue_data = self.status_data.replace(configuration)
        # status data to feature data
        feature_data = self.predictor.status2feature(try_statue_data)
        # create predicted status of this status patch
        feature_x, feature_y = self.predictor.data_split(feature_data)
        # Predict
        predicted_feature = self.predictor.predict_feature(feature_x)
        # deviation loss
        patch_array_loss = self.predictor.cal_patch_deviation(predicted_feature, feature_y)

        return np.average(patch_array_loss)


# class ProblemGA(ea.Problem):
#     def __init__(self):
#         super().__init__()
#         self.status_data: pd.DataFrame = None
#         self.predictor: CyLSTM = None
#
#         self.participle_param = toolConfig.PARAM
#         para_dict = load_param()
#         self.param_choice_dict = select_sub_dict(para_dict, self.participle_param)
#         # limitation
#         self.param_bounds = np.array([self.param_choice_dict[it]['range'] for it in list(self.param_choice_dict)])
#
#         # sub 的数据
#         self.step_unit = read_unit_from_dict(self.sub_parr_dict)
#         sub_value_range = GaMavlink.read_range_from_dict(self.sub_parr_dict)
#         # general 的数据 装欢成lstm输入的长度
#         default_value = GaMavlink.get_default_values(para_dict).loc[['default']]
#         self.segment_default_value = pd.DataFrame(default_value, dtype=np.float)
#
#         name = 'UAVProblem'  # 初始化name（函数名称，可以随意设置）boundary
#         M = 1 # 初始化M（目标维数）
#         maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
#         Dim = sub_value_range.shape[0] # 初始化Dim（决策变量维数）
#         varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
#         self.lb = sub_value_range[:, 0] / self.step_unit  # 决策变量下界
#         self.ub = sub_value_range[:, 1] / self.step_unit  # 决策变量上界
#         lbin = [0] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
#         ubin = [0] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
#         # 调用父类构造方法完成实例化
#         ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, self.lb, self.ub, lbin, ubin)
#
#     def init_bounds_and_step(self, param_bounds, step):
#         self.param_bounds = param_bounds
#         # step
#         self.step = step
#
#     def init_status(self, status_data):
#         self.status_data = status_data
#
#     def init_predictor(self, predictor):
#         self.predictor = predictor
#
#
#
#
#     def function(self, configuration):
#         configuration = self.param_value2step(configuration)
#         logging.debug(f"Optimizer configuration: {configuration}")
#         # replace parameter value
#         try_statue_data = self.status_data.replace(configuration)
#         # status data to feature data
#         feature_data = self.predictor.status2feature(try_statue_data)
#         # create predicted status of this status patch
#         feature_x, feature_y = self.predictor.data_split(feature_data)
#         # Predict
#         predicted_feature = self.predictor.predict_feature(feature_x)
#         # deviation loss
#         patch_array_loss = self.predictor.cal_patch_deviation(predicted_feature, feature_y)
#
#         return np.average(patch_array_loss)
