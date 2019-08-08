#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入相关模块
import numpy as np
from deap import base, creator, tools, algorithms


creator.create("FitnessMax", base.Fitness, weights=(1,1))

class Policy(np.ndarray):
    goal = np.array([[1,2,3],[4,0,6],[7,5,8]])
    state = np.array([[2,4,3],[1,0,6],[7,5,8]])

    def __new__(cls, *args, **kwargs):
        obj = super(Policy, cls).__new__(cls, *args, **kwargs)
        obj.fitness = creator.FitnessMax()
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.fitness = getattr(obj, 'fitness', creator.FitnessMax())
        self.fitness=creator.FitnessMax()

    @staticmethod
    def random():
        k = np.random.randint(5,10)
        obj = np.random.randint(0,9,k).view(Policy)
        obj.fitness=creator.FitnessMax()
        return obj

    def result(self, state):
        state = state.copy()
        for a in self:
            if a==0:
                state[0,0], state[0,1],state[1,0]=state[0,1],state[1,0],state[0,0]
            elif a==1:
                state[0,0], state[0,1],state[1,0]=state[1,0],state[0,0],state[0,1]
            elif a==2:
                state[0,1], state[0,2],state[1,2]=state[0,2],state[1,2],state[0,1]
            elif a==3:
                state[0,1], state[0,2],state[1,2]=state[1,2],state[0,1],state[0,2]
            elif a==4:
                state[1,0], state[2,0],state[2,1]= state[2,0],state[2,1],state[1,0]
            elif a==5:
                state[1,0], state[2,0],state[2,1]= state[2,1],state[1,0],state[2,0]
            elif a==6:
                state[2,1], state[2,2],state[1,2]= state[2,2],state[1,2],state[2,1]
            elif a==7:
                state[2,1], state[2,2],state[1,2]= state[1,2], state[2,1], state[2,2]
            else:
                pass
        return state

    def evaluate(self):
        state = self.result(Policy.state)
        v =0
        for k in range(3):
            for l in range(3):
                if state[k,l]==Policy.goal[k,l]:
                    v +=1
        return v, (self!=8).sum()


# 构造工具箱
toolbox = base.Toolbox()
toolbox.register("individual", Policy.random)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=8, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", Policy.evaluate)

# 创建种群，执行遗传算法
pop = toolbox.population(n=100)
algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.5, mutpb=0.1, ngen=350, verbose=False)
ind = tools.selBest(pop,1)[0]
print(f'''
最优分类: {ind} 
适应度: {ind.fitness.values}
''')
