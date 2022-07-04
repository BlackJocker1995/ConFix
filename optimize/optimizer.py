import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import geatpy as ea

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
        configuration = minimize(self.problem.function, self.start_value, bounds=self.param_bounds,
                                 method='nelder-mead',
                                 options={'xatol': 1e-6, 'disp': False, 'maxiter': 20})
        configuration = self.problem.param_value2step(configuration.final_simplex[0][0])
        return configuration


class GAOptimizer(DroneOptimizer):
    def __init__(self):
        super().__init__()

    def start_optimize(self):
        NINDs = 3000
        Encoding = 'RI'  # 编码方式
        Field = ea.crtfld(Encoding, self.problem.varTypes, self.problem.ranges,
                          self.problem.borders)  # 创建区域描述器
        population = ea.Population(Encoding, Field, NINDs) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        # 自定义初始化的种群 moea_NSGA2_templet
        """===============================算法参数设置============================="""
        self.algorithm = ea.moea_NSGA2_templet(self.problem, population)  # 实例化一个算法模板对象
        self.algorithm.MAXGEN = 300 # 最大进化代数
        self.algorithm.mutOper.Pm = 0.5  # 修改变异算子的变异概率
        self.algorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
        self.algorithm.maxTrappedCount = 10
        self.algorithm.drawing = 1#
        """==========================调用算法模板进行种群进化======================="""
        [NDSet, population]  = self.algorithm.run()

        with open('NDSetnew.pkl','wb') as f:
            pickle.dump(NDSet, f)
        NDSet.save()  # 把非支配种群的信息保存到文件中

        ea.moeaplot(NDSet.ObjV, xyzLabel=['No. of Solutions Covered by Range', 'Safe/Pass Ratio of Covered Solutions'])

        # 输出
        print('用时：%s 秒' % (self.algorithm.passTime))
        print('非支配个体数：%s 个' % (NDSet.sizes))
        print('单位时间找到帕累托前沿点个数：%s 个' % (int(NDSet.sizes // self.algorithm.passTime)))

        # 计算指标
        PF = self.problem.getReferObjV()  # 获取真实前沿，详见Problem.py中关于Problem类的定义
        if PF is not None and NDSet.sizes != 0:
            GD = ea.indicator.GD(NDSet.ObjV, PF)  # 计算GD指标
            IGD = ea.indicator.IGD(NDSet.ObjV, PF)  # 计算IGD指标
            HV = ea.indicator.HV(NDSet.ObjV, PF)  # 计算HV指标
            Spacing = ea.indicator.Spacing(NDSet.ObjV)  # 计算Spacing指标
            print('GD', GD)
            print('IGD', IGD)
            print('HV', HV)
            print('Spacing', Spacing)
        """============================进化过程指标追踪分析==========================="""
        if PF is not None:
            metricName = [['IGD'], ['HV']]
            [NDSet_trace, Metrics] = ea.indicator.moea_tracking(self.algorithm.pop_trace, PF, metricName,
                                                                self.problem.maxormins)
            # 绘制指标追踪分析图
            ea.trcplot(Metrics, labels=metricName, titles=metricName)
