# DEAP 遗传算法

[TOC]


## DEAP 进阶

### 多线程实现

### 扩展

### 实现其他模拟进化算法

#### 进化策略



#### 免疫算法

免疫算法模拟生物免疫系统的机制。这里对应于变量的"个体”被称为抗体。目标函数不再是个体的适应度而是抗体与抗原的亲和力。随机选择抗体依赖于浓度的概念，在大小为$N$的抗体种群中，抗体$X_i$的浓度为
$$
C_{X_i}=\frac{\sharp[X_i]_\epsilon}{N},
$$
其中$[X_i]_\epsilon=\{X_j|a(X_i,X_j)>\epsilon\}$，$a(X_i,X_j)$表示$X_i,X_j$的亲和度（编码相似性）。

**算法**

1. 初始化：随机产生$N_1$个抗体$X_1(0),\cdots ,X_{N_1}(0)$，并从记忆库中选取$N_2$个抗体$X_{N_1+1}(0),\cdots ,X_{N_2}(0)$，（若记忆库为空，则随机选择）构成初始种群$X(0)$。

2. 对每个第$t$代抗体$X_i(t)$，计算

   - 浓度$C_{X_i(t)}$
   - 与抗原的亲和度$J(X_i(t))$，一般是目标函数值
   - 增殖概率$P(X_i(t))\sim C_{X_i(t)}/J(X_i(t))$ (标准化)

   以分布$\{P(X_i(t))\}$从$X(t)$中选择$N_1$个抗体作为母本，和$N_2$个抗体存入记忆库。对母本进行一次遗传操作(变异、杂交)，再连同记忆库中的抗体构成$t+1$代抗体种群。

3. 重复2直到满足终止条件。

免疫算法和遗传算法其实没有多大不同，是遗传算法的改进：削弱浓度高的抗体的增殖概率，同时创建一个记忆库与母本隔离，保留抗体多样性。





## DEAP 复杂应用

### 融合智能局部搜索算法

这或许是GA最有意义的话题。如果遗传算法是受Darwin进化论的启发，那么局部搜索的GA是受Lamarck理论的启发。局部搜索模仿的就是生物后天学习能力，这种能力连同基因共同决定生物是否能够适应环境。变异算子有一定的局部搜索能力，但是不具有确定性，而且也容易跳出局部。而像最速下降法那样的局部搜索算法通常能保证收敛到局部最优解。

#### 局部搜索的GA的基本框架

按照生物学基本认识，生物后天习得的技能不能直接遗传给后代。对于普通生物来说，没有教学过程；而人类是可以把父辈的知识直接教授给后代。我们可以设计出两种方案。

*普通生物GA*

- 设置初始种群，循环执行下述操作
  - 对种群进行遗传操作
  - 以每个个体作为初始值进行局部搜索
  - 将最终搜索到的结果对应的目标函数值作为个体的适应值
- 最后一代进行一次局部搜索，获得最优解。



*高等生物GA*

- 设置初始种群，循环执行下述操作
  - 对种群进行遗传操作
  - 以每个个体作为初始值进行局部搜索
  - 将搜索最终结果作为新的个体，构成新的种群（计算新个体适应值以判断是否收敛）

普通版本，其实只是引入了一种新的计算适应值的方法，并没有改变GA基本框架。而高等版本，实际上延伸了遗传操作，嵌入了新的变异算子。设计这种算法需要了解DEAP的底层实现。



## DEAP 进阶

### 多线程实现

### 扩展

## DEAP 复杂应用

### 用DEAP实现其他随机算法

DEAP虽然是为遗传算法（以及多种进化算法）创建的框架，但完全可以用于实现其他随机算法。我们需要做的是重组一些遗传算法中的组件。

#### 模拟退火算法

**模拟退火算法**

1. 设置温度$T=T_0$，当前转态$s$，

2. 循环执行下述过程，直到某个给定条件

   - 计算出下一步的状态$s'$

   - 执行转态转移：

     $p=\min\{1,e^{-\Delta/T}\}, \Delta=f(s')-f(s)$

     按照Metropolis 准则更新状态$s:=\begin{cases}s',p\\s,1-p\end{cases}$

3. 降温，如$T:=T*c^n, c=0.99$。如果没有达到停止条件，那么回到2。

**注** 当$\Delta<0$时，必有$s:=s'$，其余按照Metropolis 准则更新状态。

```python
import math, random

from deap import base, creator, tools, algorithms

def move(state, toolbox, T):
    """Transition of states
    """
    new_state = toolbox.clone(state)
    new_state, = toolbox.get_neighbour(new_state)
    new_state.fitness.values = toolbox.evaluate(new_state)
    D = new_state.fitness.values[0] - state.fitness.values[0]
    return metropolis(state, new_state, D, T)

def metropolis(state, new_state, D, T):
    """Metropolis rule
    """
    if D > 0:
        p = min((1, math.exp(-D/T)))
        if random.random() <= p:
            return new_state
        else:
            return state
    else:
        return new_state

class Annealing:
    """Simulated Annealing algorithm
    """

    c = 0.99
    cc = 0.999
    nepoch = 50

    def __call__(self, state, toolbox, initT, ngen, stats=None, verbose=__debug__):
        """Simulated Annealing algorithm
        
        Arguments:
            state {list|array} -- state of the physical body in annealing
            toolbox {Toolbox} -- toolbox of DEAP
            initT {number} -- initial temperature
            ngen {int} -- number of generation
        
        Keyword Arguments:
            stats, verbose -- the same to GA
        
        Returns:
            state, logbook
        """

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the states with an invalid fitness
        if not state.fitness.valid:
            state.fitness.values = toolbox.evaluate(state)

        record = stats.compile([state]) if stats else {}
        logbook.record(gen=0, nevals=1, **record)
        if verbose:
            print(logbook.stream)

        # Begin the annealing process
        v = state.fitness.values
        for gen in range(1, ngen + 1):
            T = initT
            init = state[:]
            for epoch in range(Annealing.nepoch):
                new_state = move(state, toolbox, T)
                state = new_state
                T *= Annealing.cc ** epoch
            initT *= Annealing.c ** gen
            # Append the current state statistics to the logbook
            record = stats.compile([state]) if stats else {}
            logbook.record(gen=gen, nevals=1, **record)
        return state, logbook

annealing = Annealing()
```

本程序允许在内层循环中控制温度；统计信息只记录内层循环的最终结果，没有记录过程。读者也可以设计自己的改进方案。我们来看一个样例。

```python
from sa import *
# 导入其他模块

# 定义一个线性回归问题，大小为3 X 10
IND_SIZE = 3
N = 10
x = np.random.random(IND_SIZE)
A = 10*np.random.random((N, IND_SIZE))
b = A @ x + np.random.random(N)/100

def evaluate(state):
    return LA.norm(A @ state-b),


creator.create("FitnessMin", base.Fitness, weights=(1,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("gene", np.random.random)
toolbox.register("state", tools.initRepeat, creator.Individual,
                 toolbox.gene, n=IND_SIZE)

# 获取邻解，按照DEAP变异算子的语法设计
toolbox.register("get_neighbour", tools.mutGaussian, mu=0, sigma=0.1, indpb=1)
toolbox.register("evaluate", evaluate)

stats = tools.Statistics(key=lambda s: s.fitness.values)
stats.register("value", lambda x: x[0])

# 运行模拟退火算法
s = toolbox.state()
s, logbook = annealing(s, toolbox=toolbox, initT=10, ngen=100, stats=stats, verbose=False)

print(f'满意解(误差): {s} ({s.fitness.values[0]})')
print(f'真实最优解(误差): {x} ({LA.norm(A @ x-b)})')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['simhei'] 
matplotlib.rcParams['font.family']='sans-serif'
gen, value = logbook.select("gen", "value")
fig, ax = plt.subplots()
line = ax.plot(gen, value, "b-", label="误差值")
ax.set_xlabel("循环次数")
ax.set_ylabel("误差值", color="b")
ax.set_title("线性回归问题：循环次数-误差关系图")
for tl in ax.get_yticklabels():
    tl.set_color("b")
plt.show()


'''输出：
满意解(误差): [0.52922139 0.50438348 0.55431677] (0.12253723534316484)
真实最优解(误差): [0.52630209 0.51057957 0.54349739] (0.010955574031229948)
'''
```



### 融合智能局部搜索算法

这或许是GA最有意义的话题。如果遗传算法是受Darwin进化论的启发，那么局部搜索的GA是受Lamarck理论的启发。

## DEAP 源码解读



## 其他遗传算法框架

国内有一个出色的GA框架https://github.com/PytLab/gaft。这个框架的设计更符合一般的OOP风格，没有特别复杂的设计模式。

### 其他框架介绍

### 自制框架


