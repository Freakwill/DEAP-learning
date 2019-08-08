#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入相关模块
import numpy as np
import scipy.spatial.distance
from deap import base, creator, tools, algorithms

from utils import cxTwoPointCopy

# 描述背包问题
from sklearn.datasets import make_blobs

n_samples = 200
random_state = 170
X, _ = make_blobs(n_samples=n_samples, random_state=random_state)
K=3

D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))

def evaluate(individual):
    # 返回值: 是否满足条件, 目标函数
    W = 0
    for k in range(K):
        ck = [g==k for g in individual]
        Dk = D[ck, :][:, ck]
        W += Dk.sum()
    return W,

# 工具箱构造
IND_SIZE = X.shape[0]
creator.create("FitnessMin", base.Fitness, weights=(-1,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("gene", np.random.randint, 0, K)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.gene, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=K-1, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 创建种群，执行遗传算法
pop = toolbox.population(n=100)
algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.6, mutpb=0.17, ngen=150, verbose=False)
algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.5, mutpb=0.13, ngen=150, verbose=False)
algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.2, mutpb=0.1, ngen=50, verbose=False)
ind = tools.selBest(pop,1)[0]
print(f'''
最优分类: {ind} 
能量: {ind.fitness.values[0]}
''')


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['simhei'] 
matplotlib.rcParams['font.family']='sans-serif'

for k, c in enumerate(['r','b', 'g']):
    Ck = X[[g==k for g in ind],:]
    plt.scatter(Ck[:,0],Ck[:,1], c=c)
    plt.title(f'GA 均值聚类 ({n_samples}个样本)')
plt.show()