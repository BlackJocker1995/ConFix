import numpy as np
import pandas as pd
from scipy.optimize import minimize

from Cptool.config import toolConfig
from Cptool.mavtool import load_param, select_sub_dict
from ModelFit.approximate import CyLSTM
from optimize.problem import Problem, DTWLossProblem


class DroneOptimizer:
    def __init__(self, problem_name="DTW"):
        self.predictor: CyLSTM = None
        if problem_name == "DTW":
            self.problem = DTWLossProblem()
        else:
            self.problem = Problem()
        self.start_value = None

        self.participle_param = toolConfig.PARAM
        para_dict = load_param()
        self.param_choice_dict = select_sub_dict(para_dict, self.participle_param)
        # limitation
        self.param_bounds = np.array([self.param_choice_dict[it]['range'] for it in list(self.param_choice_dict)])

    def set_status(self, status_data):
        self.problem.init_status(status_data)
        current_param_value: pd.DataFrame = status_data[self.participle_param]
        current_param_value = current_param_value.drop_duplicates(keep="first")
        self.start_value = current_param_value.to_numpy()[0]

    def set_predictor(self, predictor):
        self.problem.init_predictor(predictor)

    def set_bounds(self):
        # step
        step = np.array([self.param_choice_dict[it]['step'] for it in list(self.param_choice_dict)])
        self.problem.init_bounds_and_step(self.param_bounds, step)

    def start_optimize(self):
        pass


class AdamGradient(DroneOptimizer):
    def __init__(self):
        super().__init__()

    def start_optimize(self):
        res = minimize(self.problem.function, self.start_value, method='nelder-mead',
                       options={'xatol': 1e-8, 'disp': True})
        print(res)
