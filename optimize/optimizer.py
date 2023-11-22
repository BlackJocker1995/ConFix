from collections import deque
from random import random

import geatpy as ea
import numpy as np
import pandas as pd
import pyswarms as ps
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from Cptool.config import toolConfig
from Cptool.mavtool import load_param, select_sub_dict, read_unit_from_dict, read_range_from_dict, get_default_values
from ModelFit.approximate import CyLSTM
from optimize.problem import Problem, ProblemGA, ProblemDQN, ProblemPSO


class DroneOptimizer:
    def __init__(self):
        self.predictor: CyLSTM = None
        self.problem = Problem()
        self.start_value = None

        self.participle_param = toolConfig.PARAM_PART
        para_dict = load_param()

        # default value, step and boundary
        self.param_choice_dict = select_sub_dict(para_dict, self.participle_param)
        self.step_unit = read_unit_from_dict(self.param_choice_dict)
        self.default_pop = (get_default_values(self.param_choice_dict) / self.step_unit).to_numpy(dtype=int)
        self.sub_value_range = read_range_from_dict(self.param_choice_dict)
        # boundary limitation

    def set_status(self, status_data):
        current_param_value: pd.DataFrame = status_data[self.participle_param]
        current_param_value = current_param_value.drop_duplicates(keep="first")
        self.start_value = current_param_value.to_numpy()[0]
        self.problem.init_status(status_data)

    def set_predictor(self, predictor):
        self.problem.init_predictor(predictor)

    def set_bounds(self):
        # step
        step = np.array([self.param_choice_dict[it]['step'] for it in list(self.param_choice_dict)])
        self.problem.init_step(step)

    def start_optimize(self):
        pass


class GAOptimizer(DroneOptimizer):
    def __init__(self):
        super(GAOptimizer, self).__init__()

        name = 'UAVProblem'  # 初始化name（函数名称，可以随意设置）boundary
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = self.sub_value_range.shape[0]  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = self.sub_value_range[:, 0] // self.step_unit  # Lower bound for decision variables
        ub = self.sub_value_range[:, 1] // self.step_unit  # Upper bound for decision variables
        lbin = [1] * Dim  # Include lower bound
        ubin = [1] * Dim  # Include upper bound

        # 调用父类构造方法完成实例化
        self.problem = ProblemGA(name=name, M=M, maxormins=maxormins, Dim=self.sub_value_range.shape[0],
                                 varTypes=varTypes, lb=lb, ub=ub, lbin=lbin, ubin=ubin)

    def start_optimize(self):
        NINDs = 40
        Encoding = 'RI'  # 编码方式
        Field = ea.crtfld(Encoding, self.problem.varTypes, self.problem.ranges,
                          self.problem.borders)  # 创建区域描述器
        population = (ea.Population(Encoding, Field, NINDs))  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        # 自定义初始化的种群soea_DE_currentToBest_1_bin_templet
        """===============================算法参数设置============================="""
        self.algorithm = ea.soea_DE_currentToBest_1_bin_templet(self.problem, population)  # 实例化一个算法模板对象
        self.algorithm.MAXGEN = 80  # 最大进化代数
        self.algorithm.mutOper.F = 0.7  # 差分进化中的参数F
        self.algorithm.recOper.XOVR = 0.7  # 重组概率
        self.algorithm.trappedValue = 0.1  # “进化停滞”判断阈值
        self.algorithm.maxTrappedCount = 10  # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
        self.algorithm.drawing = 0  #
        """===========================根据先验知识创建先知种群======================="""
        # prophetChrom = np.array(self.default_pop // self.step_unit, dtype=int) # 假设已知为一条比较优秀的染色体
        prophetChrom = np.array([self.start_value // self.step_unit], dtype=int)
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


class SwarmOptimizer(DroneOptimizer):
    def __init__(self):
        super().__init__()

        name = 'SwarmProblem'
        self.lb = self.sub_value_range[:, 0] - 0.1  # Lower bound for decision variables
        self.ub = self.sub_value_range[:, 1] + 0.1 # Upper bound for decision variables
        self.dim = self.sub_value_range.shape[0]  # 初始化Dim（决策变量维数）

        self.problem = ProblemPSO(name)

    def start_optimize(self, init_pos=False):
        pop = 20  # Population Size
        options = {'c1': 0.7, 'c2': 0.7, 'w': 0.8}
        if init_pos:
            optimizer = ps.single.GlobalBestPSO(n_particles=pop, dimensions=self.dim,
                                            options=options, bounds=(self.lb, self.ub),
                                            init_pos=np.repeat(self.start_value.reshape(1, -1), pop, axis=0))
        else:
            optimizer = ps.single.GlobalBestPSO(n_particles=pop, dimensions=self.dim,
                                                options=options, bounds=(self.lb, self.ub))
        # Perform optimization
        best_cost, best_pos = optimizer.optimize(self.problem.function_swarm, verbose=False, iters=30)
        configuration = self.problem.param_value2pd_single(best_pos)
        return configuration


class DQNOptimizer(DroneOptimizer):
    def __init__(self, state_size, action_size):
        super(DQNOptimizer, self).__init__(state_size, action_size)
        self.problem = ProblemDQN()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation="softmax", name="fc2"))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        act_values = self.model.predict(state)
        act_values * self.problem.step
        return act_values  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def start_optimize(self):
        # TODO: optimize start
        pass
