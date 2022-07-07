import geatpy as ea
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from Cptool.config import toolConfig
from Cptool.mavtool import load_param, select_sub_dict, read_unit_from_dict, read_range_from_dict
from ModelFit.approximate import CyLSTM
from optimize.problem import Problem, DTWLossProblem, ProblemGA


class DroneOptimizer:
    def __init__(self):
        self.predictor: CyLSTM = None
        self.problem = Problem()
        self.start_value = None

        self.participle_param = toolConfig.PARAM
        para_dict = load_param()
        self.param_choice_dict = select_sub_dict(para_dict, self.participle_param)
        # limitation
        self.param_bounds = np.array([self.param_choice_dict[it]['range'] for it in list(self.param_choice_dict)])

    def set_status(self, status_data):
        current_param_value: pd.DataFrame = status_data[self.participle_param]
        current_param_value = current_param_value.drop_duplicates(keep="first")
        self.start_value = current_param_value.to_numpy()[0]
        self.problem.init_status(status_data, self.start_value)

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
        self.problem = DTWLossProblem()

    def start_optimize(self):
        configuration = minimize(self.problem.function, self.start_value, bounds=self.param_bounds,
                                 method='nelder-mead',
                                 options={'xatol': 1e-2, 'disp': False, 'maxiter': 20})
        configuration = self.problem.param_value2step(configuration.x)
        return configuration


class GAOptimizer(DroneOptimizer):
    def __init__(self):
        super().__init__()

        self.participle_param = toolConfig.PARAM
        para_dict = load_param()
        self.param_choice_dict = select_sub_dict(para_dict, self.participle_param)

        # sub 的数据
        self.step_unit = read_unit_from_dict(self.param_choice_dict)
        sub_value_range = read_range_from_dict(self.param_choice_dict)

        name = 'UAVProblem'  # 初始化name（函数名称，可以随意设置）boundary
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = sub_value_range.shape[0]  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = sub_value_range[:, 0] // self.step_unit  # 决策变量下界
        ub = sub_value_range[:, 1] // self.step_unit  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

        # 调用父类构造方法完成实例化
        self.problem = ProblemGA(name=name, M=M, maxormins=maxormins, Dim=sub_value_range[0],
                                 varTypes=varTypes, lb=lb, ub=ub, lbin=lbin, ubin=ubin)

    def start_optimize(self):
        NINDs = 50
        Encoding = 'RI'  # 编码方式
        Field = ea.crtfld(Encoding, self.problem.varTypes, self.problem.ranges,
                          self.problem.borders)  # 创建区域描述器
        population = (ea.Population(Encoding, Field, NINDs))  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        # 自定义初始化的种群soea_DE_currentToBest_1_bin_templet
        """===============================算法参数设置============================="""
        self.algorithm = ea.soea_DE_currentToBest_1_bin_templet(self.problem, population)  # 实例化一个算法模板对象
        self.algorithm.MAXGEN = 50  # 最大进化代数
        self.algorithm.mutOper.F = 0.7  # 差分进化中的参数F
        self.algorithm.recOper.XOVR = 0.7  # 重组概率
        self.algorithm.trappedValue = 0.1  # “进化停滞”判断阈值
        self.algorithm.maxTrappedCount = 10  # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
        self.algorithm.drawing = 0  #
        """===========================根据先验知识创建先知种群======================="""
        prophetChrom = np.array([self.start_value // self.step_unit], dtype=int)  # 假设已知为一条比较优秀的染色体
        prophetPop = ea.Population(Encoding, Field, 1, prophetChrom)  # 实例化种群对象（设置个体数为1）

        self.algorithm.call_aimFunc(prophetPop)
        """==========================调用算法模板进行种群进化======================="""
        [self.NDSet, self.population] = self.algorithm.run(prophetPop)

        obj_trace = np.array(self.population.Phen)
        var_trace = np.array(self.population.ObjV)

        # 去除重复
        candidate_var_index = np.unique(var_trace, axis=0, return_index=True)[1]
        candidate_var = var_trace[candidate_var_index].reshape(-1)
        candidate_obj = obj_trace[candidate_var_index]

        candidate = self.problem.maxormins * candidate_var
        # 从小到大
        candidate_index = np.argsort(candidate)
        candidate_obj = candidate_obj[candidate_index]

        return self.problem.param_value2step(candidate_obj)

