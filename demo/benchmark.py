#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入相关模块
import numpy as np
import numpy.linalg as LA
from deap import base, creator, tools, algorithms

from utils import cxTwoPointCopy

import vga

# 定义适应值类，作为base.Fitness子类，包含weights属性
creator.create("FitnessMin", base.Fitness, weights=(-1,))
# 定义个体类，作为np.ndarray子类，包含fitness属性
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

"""定义工具箱：
1. 注册构造个体、总群的方法
2. 定义适应值函数
3. 定义并注册遗传操作函数
"""
IND_SIZE = 25
toolbox = base.Toolbox()
toolbox.register("gene", np.random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.gene, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 采用DEAP提供的遗传操作函数：交叉、变异、选择
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

N = 1000
a = np.random.random(N)
A = np.random.random((N, IND_SIZE))
def evaluate(individual):
    """
    计算个体适应值
    
    参数
        individual: {Individual} -- 代表个体的数组
    返回
        元组，与weights匹配
    """
    return LA.norm(A @ individual-a),
toolbox.register("evaluate", evaluate)

# 创建种群，运行遗传算法

import time

logbooks1 =  []
logbooks2 = []
d = []
for _ in range(100):
    pop = toolbox.population(n=50)
    popc = [toolbox.clone(ind) for ind in pop]

    time1 = time.perf_counter()
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    _, logbook1 = vga.eaAdaptive(pop, toolbox=toolbox, cxpb=0.7, mutpb=0.1, ngen=100, stats=stats, verbose=False)
    time2 = time.perf_counter()
    d1 = time2 - time1

    time1 = time.perf_counter()
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    _, logbook2 = algorithms.eaSimple(popc, toolbox=toolbox, cxpb=0.6, mutpb=0.1, ngen=100, stats=stats, verbose=False)

    time2 = time.perf_counter()
    d2 = time2 - time1
    logbooks1.append(logbook1)
    logbooks2.append(logbook2)
    d.append(d1/d2)

import matplotlib.pyplot as plt

gen = logbook1.select("gen")
fmins = np.row_stack([logbook.select("min") for logbook in logbooks1]).mean(axis=0)
# favgs = logbook1.select("avg")
# fmaxs = logbook1.select("max")
fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, fmins, "b-", label="Minimum Fitness(VGA)")
# line1 = ax1.plot(gen, favgs, "b-.", label="Average Fitness(VGA)")
line11 = ax1.plot(np.array(gen) * np.mean(d), fmins, "g-", label="Minimum Fitness(VGA) / Time")
ax1.set_xlabel("Generation / Time")
ax1.set_ylabel("Fitness")

gen = logbook2.select("gen")
fmins = np.row_stack([logbook.select("min") for logbook in logbooks2]).mean(axis=0)
# favgs = logbook2.select("avg")
# fmaxs = logbook2.select("max")

line2 = ax1.plot(gen, fmins, "r-", label="Minimum Fitness(SGA)")
# line2 = ax2.plot(gen, favgs, "r-.", label="Average Fitness(SGA)")
# line2 = ax2.plot(gen, fmaxs, "r-", label="Maximum Fitness(SGA)")

lns = line1 + line11+ line2 
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right")

plt.show()