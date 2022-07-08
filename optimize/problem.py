import logging

import geatpy as ea
import numpy as np
import pandas as pd

from Cptool.config import toolConfig
from ModelFit.approximate import CyLSTM, CyTCN
from ModelFit.config import modelConfig


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


class ProblemFunLoss(Problem):
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
        if isinstance(self.predictor, CyTCN):
            feature_y = feature_y.reshape((feature_y.shape[0], -1))
        # Predict
        predicted_feature = self.predictor.predict_feature(feature_x)
        if isinstance(self.predictor, CyTCN):
            predicted_feature = predicted_feature.reshape((predicted_feature.shape[0], -1))
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

    def aimFunc_other(self, configuration):
        x = configuration.Phen
        x = self.reasonable_range(x)

        each_loss = []
        for index, sub_param in x.iterrows():
            try_statue_data = self.status_data.replace(sub_param)
            # status data to feature data
            feature_data = self.predictor.status2feature(try_statue_data)
            # create predicted status of this status patch
            feature_x, feature_y = self.predictor.data_split(feature_data)
            # Predict
            predicted_feature = self.predictor.predict_feature(feature_x)
            # deviation loss
            patch_array_loss = self.predictor.cal_patch_deviation(predicted_feature, feature_y)

            each_loss.append(patch_array_loss)

        each_loss = np.array(each_loss).reshape((-1, 1))

        configuration.ObjV = each_loss

    def aimFunc(self, configuration):
        x = configuration.Phen
        x = self.reasonable_range(x)

        # repeat data
        repeat_status = pd.concat([self.status_data] * x.shape[0]).reset_index(drop=True)
        repeat_param = pd.DataFrame(np.repeat(x.values, self.status_data.shape[0], axis=0), columns=x.columns)
        repeat_status[toolConfig.PARAM] = repeat_param

        status_step = self.status_data.shape[0]
        feature_step = self.status_data.shape[0] - modelConfig.INPUT_LEN
        feature = pd.DataFrame()
        for i in range(x.shape[0]):
            status_index = i * status_step
            feature_index = i * feature_step
            tmp_status = repeat_status.iloc[status_index:status_index+status_step]
            # status data to feature data
            tmp_feature_data = self.predictor.status2feature(tmp_status)
            feature = pd.concat([feature, tmp_feature_data])
        # create predicted status of this status patch
        feature_x, feature_y = self.predictor.data_split(feature)
        # Predict
        predicted_feature = self.predictor.predict_feature(feature_x)
        # reshape to 3D (x number, status patch)
        predicted_feature = predicted_feature.reshape((x.shape[0], -1, predicted_feature.shape[-1]))
        feature_y = feature_y.reshape((x.shape[0], -1, predicted_feature.shape[-1]))
        # deviation loss
        patch_array_loss = self.predictor.cal_patch_deviation(predicted_feature, feature_y)

        patch_average_loss = np.average(patch_array_loss, axis=1)

        configuration.ObjV = patch_average_loss.reshape((-1, 1))

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
