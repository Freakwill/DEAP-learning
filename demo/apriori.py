#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入相关模块
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from utils import cxTwoPointCopy

# 描述问题

df = pd.read_csv('heart.csv')
df = df >0
N, p = df.shape

def C(A, B):
    if np.all(A==0) or np.all(B==0):
        return 0
    elif T(A) == 0:
        return 0
    elif np.all(B <= A):
        return 0
    return T(A | B) / T(A)

def T(J, I=None):
    if I is not None:
        J |= I
    if np.all(J==0):
        return 0
    return np.mean([np.prod([df.iloc[i,k] for k,j in enumerate(J) if j]) for i in range(N)])

t = 0.2
s = 0.8

def evaluate(ind):
    # 置信度作为主要适应度
    A, B = ind[:p], ind[p:]
    if T(A | B) > t:
        return C(A, B), T(A|B)
    else:
        return 0, T(A|B)

def evaluate(individual):
    # 返置信度作为主要适应度
    A, B = individual[:p], individual[p:]
    if T(A | B) > t:
        return min((C(A, B),s)), T(A | B)
    else:
        return 0, T(A|B)

# 构造工具箱
IND_SIZE = p * 2
creator.create("FitnessMax", base.Fitness, weights=(1,1))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("gene", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.gene, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 创建种群，执行遗传算法
pop = toolbox.population(n=20)
algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.5, mutpb=0.1, ngen=20, verbose=False)
ind = tools.selBest(pop, 1)[0]

lhs = ','.join(k for j, k in zip(ind[:p], df.keys()) if j)
rhs = ','.join(k for j, k in zip(ind[p:], df.keys()) if j)
print(f'{lhs} => {rhs} # Confidence:{C(ind[:p], ind[p:])}  Support:{T(ind[:p], ind[p:])}')
