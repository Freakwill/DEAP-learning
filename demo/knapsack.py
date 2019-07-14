#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入相关模块
import numpy as np
import numpy.linalg as LA
from deap import base, creator, tools, algorithms

# 描述背包问题
w = np.array([35,50,30,15,10,35,25,40])
c = np.array([40,60,25,20,5,60,40,25])
W = 140
def evaluate(individual):
    # 返回值: 是否满足条件, 目标函数
    return np.all(np.dot(w, individual)<=140), np.dot(c, individual)

def evaluate(individual):
    # 返回值: 是否满足条件, 目标函数
    if np.all(np.dot(w, individual)<=140):
        return np.dot(c, individual),
    else:
        return -100 * np.dot(c, individual),

# 工具箱构造
IND_SIZE = len(w)
creator.create("FitnessMax", base.Fitness, weights=(1,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("gene", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.gene, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 创建种群，执行遗传算法
pop = toolbox.population(n=50)
algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.5, mutpb=0.1, ngen=100, verbose=False)
ind = tools.selBest(pop,1)[0]
print(f'''
最优个体: {ind} 
是否满足条件: {'是' if ind.fitness.values[0] else '否'}
总价值: {ind.fitness.values[0]}
''')