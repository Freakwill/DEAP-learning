#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入相关模块
import numpy as np
import numpy.linalg as LA
from deap import base, creator, tools, algorithms

from utils import cxTwoPointCopy

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

cxpbs = np.linspace(0.1,0.9,9)

logbooks = {}
ds = {}
for cxpb in cxpbs:
    logbooks[cxpb] = []
    ds[cxpb] = []

for _ in range(100):
    pop = toolbox.population(n=50)
    for cxpb in cxpbs:
        popc = [toolbox.clone(ind) for ind in pop]
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        time1 = time.perf_counter()
        _, logbook = algorithms.eaSimple(popc, toolbox=toolbox, cxpb=cxpb, mutpb=0.1, ngen=100, stats=stats, verbose=False)
        time2 = time.perf_counter()
        d = time2 - time1
        logbooks[cxpb].append(logbook)
        ds[cxpb].append(d)


import matplotlib.pyplot as plt

fig = plt.figure()
fig.suptitle("Generation ~ Fitness")
ngen = 100

ax1 = fig.add_subplot(121)
lines1 = {}
for cxpb in cxpbs:
    time = np.mean(ds[cxpb])
    fmins = np.row_stack([logbook.select("min") for logbook in logbooks[cxpb]]).mean(axis=0)
    lines1[cxpb], = ax1.plot(np.arange(ngen+1), fmins, color=np.random.random(3), label=f"$P_c:{cxpb:.2}$")

ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness (Least Squares Error)")

labs = [lines1[cxpb].get_label() for cxpb in cxpbs]
ax1.legend([lines1[cxpb] for cxpb in cxpbs], labs, loc="center right")

ax = fig.add_subplot(122)
lines = {}
for cxpb in cxpbs:
    time = np.mean(ds[cxpb])
    fmins = np.row_stack([logbook.select("min") for logbook in logbooks[cxpb]]).mean(axis=0)
    lines[cxpb], = ax.plot(np.arange(ngen+1) * time, fmins, color=lines1[cxpb].get_color(), label=f"$P_c:{cxpb:.2}$")

ax.set_xlabel("Generation * Time")

labs = [lines[cxpb].get_label() for cxpb in cxpbs]
ax.legend([lines[cxpb] for cxpb in cxpbs], labs, loc="center right")

plt.show()