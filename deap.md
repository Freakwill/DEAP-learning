# DEAP 遗传算法

[TOC]



## 遗传算法简介

### 基本概念

#### 种群、个体、基因

#### 选择算子、进化算子

#### 模式定理

### 算法框架

#### 基本框架

#### 简单实例

#### 混合算法

### 理论分析

#### Markov 链分析

#### 公理化分析

#### 鞅分析

## DEAP 入门

本章开始介绍遗传算法框架DEAP（Distributed Evolutionary Algorithms in Python）。如官网https://deap.readthedocs.io/en/master/index.html上所说，DEAP是一种新的进化计算框架，用于快速原型和思想测试。

### 简单的例子

官网上有许多例子https://deap.readthedocs.io/en/master/examples/index.html。先来"解剖"一个例子。通过这个例子，读者就可以自行设计遗传算法，解决实际问题，同时也可以了解DEAP的设计风格。代码中的注释基大致解释了整个程序的思想。

```python
# 导入相关模块
import numpy as np
import numpy.linalg as LA
from deap import base, creator, tools, algorithms

a = np.random.random(IND_SIZE)
def evaluate(individual):
    """
    计算个体适应值
    
    参数
        individual: {Individual} -- 代表个体的数组
    返回
        元组，与weights匹配
    """
    return LA.norm(individual-a),

# 定义适应值类，作为base.Fitness子类，包含weights属性
creator.create("FitnessMin", base.Fitness, weights=(-1,))
# 定义个体类，作为np.ndarray子类，包含fitness属性
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

"""定义工具箱：
1. 注册构造个体、总群的方法
2. 定义适应值函数
3. 定义并注册遗传操作函数
"""

IND_SIZE = a.shape[0]
toolbox = base.Toolbox()
toolbox.register("gene", np.random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.gene, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 采用DEAP提供的遗传操作函数：交叉、变异、选择
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 创建种群，运行遗传算法
pop = toolbox.population(n=50)
algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.5, mutpb=0.1, ngen=1000, verbose=False)
print(f'最优个体(适应值): {ind} ({ind.fitness.values[0]})')
print(f'最优解: {a}')

""" 输出
最优个体(适应值): [0.71151324 0.69193001 0.97741192 0.45719987 0.37885289 0.01114395
 0.95605213 0.22546305 0.35582403 0.68615445] (0.020854654639729964)
最优解: [0.72821546 0.69749908 0.97231912 0.45631586 0.3778101  0.01095263
 0.94733828 0.22648515 0.36029828 0.68647196]
 """
```

这就是一个多元函数的最小值计算通用程序。读者唯一要改的，是`evaluate`函数，它应该是你的目标函数，其输入是一个`numpy.array` (Individual 是`numpy.ndarray`的子类，操作上与`numpy.array`无异)。例如，把适应值函数改成，

```python
b = np.random.random(m)
A = np.random.random((m, IND_SIZE))
def evaluate(individual):
    return LA.norm(A @ individual-b),
```

就可以解方程组$Ax=b$ (最小二乘解).

**注** 上述例子参考https://deap.readthedocs.io/en/master/overview.html



简单讲一下DEAP的设计模式。它应该采用了工厂模式。



遗传算法通常要解决一个编码问题。最常见的是把实数用二进制数组表示。实现这种转换，主要考虑小数位数。二进制是遗传算法最合适的数值表示。至少，二进制表示非常适合收敛性方面的理论分析。

### 简单应用

遗传算法是通用解法，其应用数不甚数。本节讲几个有实际意义的应用，有些有明显的实践背景、有些则有理论意义。通过这些应用，读者可以了解 DEAP 的 API的功能和用法，不必去阅读文档和源码，从中选择合适的方法去解决自己遇到的问题。

#### 背包问题

背包问题是一种常见的优化问题。人们把许多现实中复杂的问题都转化成这个直观的模型，如资本预算、货物转载和资源分配等。还有回归分析中参数的选取也是背包问题。背包问题是NP-hard问题，加上其特殊的性质，特别适合用遗传算法来解，也是各种遗传算法变体的"试金石”。

背包问题的大意是，从$n$个物品中取出若干个，放入背包中，在重量不超过背包符合的条件下，最优化总价值。设物品$j$的重量是$w_j$，价值是$c_j$，$x_j=1$表示选择物品$j$，否则$x_j=0$。建立如下优化问题。
$$
\max \sum_jc_jx_j\\
s.t. \sum_jw_jx_j\leq W, x_j=0,1
$$

```python
# 描述背包问题
w = np.array([35,50,30,15,10,35,25,40])
c = np.array([40,60,25,20,5,60,40,25])
W = 140
def evaluate(individual):
    # 返回值: 是否满足条件, 目标函数
    return np.all(np.dot(w, individual)<=W), np.dot(c, individual)

# 工具箱构造
IND_SIZE = len(w)
creator.create("FitnessMax", base.Fitness, weights=(1, 1))
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
总价值: {ind.fitness.values[1]}
''')
```

因为变异时，只需切换0-1，所以采用`tools.mutFlipBit`变异算子。

这里的适应度函数返回是一个数组，个体的适应度会从第一个分量开始比较，直到比较出大小，即所谓的"字典序”。这个适应度函数并不好，因为个体对第二个分量（总价值）不够敏感，但是有不能交换两个分量的位置（为什么？）。一种跟合理的方法是

```python
def evaluate(individual):
    # M 被设置成一个很大的数
    if np.all(np.dot(w, individual)<=W):
        return np.dot(c, individual),
    else:
        return -M * np.dot(c, individual),
```

这个适应度函数，利用了这样的尝试，当超过重量的时候，适应度应该是物品总价值的递减函数。

#### Apriori 算法

另一个可以直接用0-1编码的问题是Apriori 算法。

#### 聚类算法

背包算法也是一种聚类，把物品分成放入背包的和不放入背包的。聚类算法简单的讲就是给变量添加一个标签，而这些标签应该满足某些人类的直觉和经验。比如对平面中的点分类，应该要求不同类的点之间距离较远，而同类点之间距离较近。

设有$N$个点，第$i$个点的坐标是$x_i$, 标记为$C_i$类，定义聚类能量函数：
$$
W(C)=\frac{1}{2}\sum_k\sum_{C_i=C_j=k}d(x_i,x_j)\\
B(C)=\frac{1}{2}\sum_k\sum_{C_i\neq C_j=k}d(x_i,x_j)
$$
我们的任务是最小化$W(C)$或者最大化$B(C)$.

```python
# 导入相关模块
import itertools
import numpy as np
import numpy.linalg as LA
import scipy.spatial.distance
from deap import base, creator, tools, algorithms

# 构造数据
X = np.random.random((100,2))
K=3

def evaluate(individual):
    # 计算 W(C)
    W = 0
    for k in range(K):
        Ck = X[[g==k for g in individual],:]
        W += scipy.spatial.distance.pdist(Ck).sum()
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
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=K-1, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 创建种群，执行遗传算法
pop = toolbox.population(n=100)
algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.5, mutpb=0.1, ngen=200, verbose=False)
ind = tools.selBest(pop,1)[0]
print(f'''
最优分类: {ind} 
能量: {ind.fitness.values[0]}
''')

'''输出
最优分类: [2, 0, 1, 0, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 0, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 2, 1, 1, 2, 2, 0, 2, 0, 2, 0, 2, 2, 1, 2, 1, 1, 0, 1, 2, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 0, 2, 0, 1, 0, 2, 2, 2, 2, 2, 0, 0, 2, 1, 2, 1, 2, 0, 2, 2, 1, 2, 0, 1, 0, 0, 2, 2, 1, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 0] 
能量: 580.9665943384339
'''
```

聚类算法属于机器学习的范畴。上述程序可用scikit-learn包装一下。下面是一个可行方案。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.spatial.distance
from deap import base, creator, tools, algorithms

from sklearn.cluster import KMeans


class GAKMeans(KMeans):
    """GA for KMeans
    
    the individual of GA is the sequence of labels of samples
    the best one in last generation is the result of clustering.
    
    Extends:
        KMeans
    """
    def __init__(self, *args, **kwargs):
        super(GAKMeans, self).__init__(*args, **kwargs)
        self.n_clusters=kwargs['n_clusters']

    def config(self, X):
        # configuration for GA
        K = self.n_clusters
        D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))
        def evaluate(individual):
            W = 0
            for k in range(K):
                ck = [g==k for g in individual]
                Dk = scipy.spatial.distance.squareform(D[ck, :][:, ck])
                W += Dk.sum()
            return W,
        IND_SIZE = X.shape[0]
        creator.create("FitnessMin", base.Fitness, weights=(-1,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("gene", np.random.randint, 0, K)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.gene, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=K-1, indpb=0.01)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate)
        
        def ga():
            pop = toolbox.population(n=80)
            algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.5, mutpb=0.1, ngen=300, verbose=False)
            return tools.selBest(pop,1)[0]
        return ga

    def fit(self, X, *args, **kwargs):
        super(GAKMeans, self).fit(X, *args, **kwargs)

        best = self.config(X)()
        self.labels_ = np.array(best)
        self.cluster_centers_= np.vstack([X[[g==k for g in self.labels_], :].mean(axis=0) for k in range(self.n_clusters)])
        self.inertia_ = best.fitness.values[0]
        return self

```



#### 神经网络结构优化



#### 多元时间序列预测



#### 参数选择





### 算法细节改进

`algorithms.eaSimple`是遗传算法最常规的实现。它大致的流程是：


- 计算初始种群中个体适应值
- 对每一代循环:
    - 选择下一代总群
    - 遗传操作
    
      `offspring = varAnd(offspring, toolbox, cxpb, mutpb)`
    - 计算个体适应值并更新总群

这里的核心就是`varAnd`函数。读者要设计自己的遗传算法，可以重新实现这个函数。



### 算法可视化




## DEAP 进阶

### 多线程实现

### 扩展

## DEAP 复杂应用

### 融合智能局部搜索算法

## DEAP 源码解读



## 其他遗传算法框架

### 其他框架介绍

### 自制框架


