#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入相关模块
import numpy as np
import numpy.linalg as LA
from deap import base, creator, tools, algorithms

# 定义适应值类，作为base.Fitness子类，包含weights属性
creator.create("FitnessMin", base.Fitness, weights=(-1,))
# 定义个体类，作为np.ndarray子类，包含fitness属性
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

"""定义工具箱：
1. 注册构造个体、总群的方法
2. 定义适应值函数
3. 定义并注册遗传操作函数
"""
IND_SIZE = 10
toolbox = base.Toolbox()
toolbox.register("gene", np.random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.gene, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 采用DEAP提供的遗传操作函数：交叉、变异、选择
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

x = np.random.random(IND_SIZE)
def evaluate(individual):
    """
    计算个体适应值
    
    参数
        individual: {Individual} -- 代表个体的数组
    返回
        元组，与weights匹配
    """
    return LA.norm(individual-x),
toolbox.register("evaluate", evaluate)

# 创建种群，运行遗传算法
pop = toolbox.population(n=50)
algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.5, mutpb=0.1, ngen=1000, verbose=False)
print(f'最优个体(适应值): {ind} ({ind.fitness.values[0]})')
print(f'最优解: {x}')

""" 输出
最优个体(适应值): [0.71151324 0.69193001 0.97741192 0.45719987 0.37885289 0.01114395
 0.95605213 0.22546305 0.35582403 0.68615445] (0.020854654639729964)
最优解: [0.72821546 0.69749908 0.97231912 0.45631586 0.3778101  0.01095263
 0.94733828 0.22648515 0.36029828 0.68647196]
 """