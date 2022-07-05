import logging

import geatpy as ea
import numpy as np
import pandas as pd

from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM


class Problem():
    def __init__(self):
        self.status_data: pd.DataFrame = None
        self.predictor: CyLSTM = None
        self.param_bounds = None
        self.start_value = None
        self.step = None

    def init_status(self, status_data, start_value):
        self.status_data = status_data
        self.start_value = start_value

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


class ProblemGA(ea.Problem, Problem):
    def __init__(self, name, M, maxormins, Dim,
                 varTypes, lb, ub, lbin, ubin):
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)

        self.status_data: pd.DataFrame = None
        self.predictor: CyLSTM = None
        self.start_value = None

        self.sensor_data = None
        self.param_data = None

    def init_status(self, status_data, start_value):
        self.status_data = status_data
        self.start_value = start_value

        self.sensor_data = status_data[toolConfig.STATUS_ORDER]
        self.param_data = status_data[toolConfig.PARAM]

    def aimFunc(self, configuration):
        x = configuration.Phen
        x = self.reasonable_range(x)

        each_loss = []
        for sub_param in x.iterrows():
            try_statue_data = self.status_data.replace(x)
            # status data to feature data
            feature_data = self.predictor.status2feature(try_statue_data)
            # create predicted status of this status patch
            feature_x, feature_y = self.predictor.data_split(feature_data)
            # Predict
            predicted_feature = self.predictor.predict_feature(feature_x)
            # deviation loss
            patch_array_loss = self.predictor.cal_patch_deviation(predicted_feature, feature_y)

            sub_loss = np.average(patch_array_loss)

            each_loss.append(sub_loss)

        each_loss = np.array(each_loss).reshape((-1, 1))

        configuration.ObjV = each_loss

    def param_value2step(self, configuration):
        np_config = configuration * self.step
        np_config = pd.DataFrame(np_config, columns=toolConfig.PARAM)
        return np_config.iloc[0].to_dict()

    def reasonable_range(self, param):
        """
        还原数据
        :param param:
        :return:
        """
        np_config = param * self.step
        np_config = pd.DataFrame(np_config, columns=toolConfig.PARAM)
        return np_config
