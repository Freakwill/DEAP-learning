#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入相关模块
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

from utils import cxTwoPointCopy, clean, unique

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

def T(J):
    if np.all(J==0):
        return 0
    return np.mean([np.prod([df.iloc[i,k] for k,j in enumerate(J) if j]) for i in range(N)])

t = 0.2
creator.create("FitnessMax", base.Fitness, weights=(1,1))

class Rule:
    '''Association Rule
    '''
    keys = df.keys()
    def __init__(self, A, B, keys=None):
        self.A = A
        self.B = B
        self.fitness = creator.FitnessMax()

    def normalize(self):
        # make A & B == 0
        for k, a in enumerate(self.A):
            if 0<a == self.B[k]:
                self.B[k]=0

    @property
    def cardA(self):
        return np.sum(self.A==1)

    @property
    def cardB(self):
        return np.sum(self.B==1)

    def __eq__(self, other):
        return np.all(self.A == other.A) and np.all(self.B == other.B)

    def __lt__(self, other):
        return self.evaluate() == other.evaluate() and (np.all(self.A< other.A) and np.all(other.B<=self.B) or 
            np.all(self.A<= other.A) and np.all(other.B<self.B))

    @classmethod
    def random(cls, p):
        A = np.random.randint(0, 2, p)
        B = np.random.randint(0, 2, p)
        for k in range(p):
            if 0< A[k] == B[k]:
                B[k]=0
        return cls(A, B)

    @property
    def support(self):
        return T(self.A|self.B)

    @property
    def confidence(self):
        return C(self.A, self.B)

    def evaluate(self):
        if self.support > t:
            return self.confidence, self.support
        else:
            return 0, self.support

    def mate(self, other):
        self.A, other.A = cxTwoPointCopy(self.A, other.A)
        self.B, other.B = cxTwoPointCopy(self.B, other.B)
        return self, other

    def mutate(self, indpb):
        self.A, = tools.mutFlipBit(self.A, indpb=indpb)
        self.B, = tools.mutFlipBit(self.B, indpb=indpb)
        return self,

    def __str__(self):
        self.normalize()
        lhs = ','.join(k for j, k in zip(self.A, self.keys) if j)
        rhs = ','.join(k for j, k in zip(self.B, self.keys) if j)
        return  ' => '.join((lhs, rhs))


# 工具箱构造
toolbox = base.Toolbox()
import multiprocessing
pool = multiprocessing.Pool()
toolbox.register("map", pool.map)
toolbox.register("individual", Rule.random, p)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", Rule.mate)
toolbox.register("mutate", Rule.mutate, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", Rule.evaluate)

# 创建种群，执行遗传算法
pop = toolbox.population(n=30)
algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.5, mutpb=0.1, ngen=20, verbose=False)

best = unique(tools.selBest(pop, None))
for ind in best:
    if ind.support > t:
        print(ind, ind.confidence, ind.support)

print('去掉多余规则')
for ind in clean(best):
    if ind.support > t:
        print(ind, ind.confidence, ind.support)
