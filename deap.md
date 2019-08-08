# DEAP 遗传算法

[TOC]



## 遗传算法简介

遗传算法是一种通用优化算法。它的发明受Darwin进化论的启发。进化论被认为能解释任何生物发展、生灭的一般理论。遗传算法模拟进化机制，原则上可用于任何优化问题。使用这种算法，你并不需要专业领域知识。效率方面，它并不能和专业化的算法相比。但是，它的一些特点是令人印象深刻的。

1. 和专业化算法结合，可以提高算法效率
2. 在专业化算法缺失的情况下，可以直接使用
3. 使用者不需要用专业领域知识为算法设定规则。
4. 本质上是自适应的并行计算

### 基本概念

本书是给那些已经了解过GA的读者看的。一些基本的概念不会被详细说明，只是约定一下符号；本书是技术类书籍，对书中的定理也不会给出证明。

#### 种群、个体、基因

遗传算法复制了进化论的基本概念。在遗传算法的体系里，个体完全被一串"基因"编码，也就是染色体。因此我们并不严格区分个体和染色体。种群则是这些个体的集合，或者序列。

一个种群里可以有若干个相同个体，但也可以认为是不同个体，只是拥有相同的染色体。

**定义（种群、个体、基因）**

设$\Gamma$代表基因集合，则个体空间$H=\Gamma^l$，而包含$m$个个体的种群空间是$H^m$。这里面假定染色体的每一位基因都选自$\Gamma $。如果不要求每一位基因都用同一个集合，那么令个体空间$H=\prod_i\Gamma_i$，其中$\Gamma_i$是染色体第$i$位的基因集合。

通常染色体有固定长度$l$，如果不确定，则个体空间可定义为$\Gamma^*$，$\Gamma$的Kleene闭包。

**例**

人类的染色体可以用由四个碱基$\{G,A,T,C\}$生成的字符串表示。这四个碱基可以作为遗传算法的基因。不过生物学上的基因是指有遗传效应的DNA片段。

**例**

在设计遗传算法的时候，编码是至关重要的一步。GA最常用的是二进制编码。若对实数域上的函数最优化，则可把单个实数作为个体，而把它的二进制形式向量化后作为个体的编码。先设定好统一的小数位，如4。再把每个数，如7.5表示成近似的4位二进制数，111.1000，在长度为8的编码中就是01111000。这种编码方案的误差控制在$2^{-4}$以内。

上面的编码是有缺点的，如果函数定义域不是$[0,2^4]$，那么这种编码就没有效率，可能太窄不能编码某些实数，就是太广编码了定义域以外的实数。有一种二进制编码方案可以解决克服这个缺点。把区间$[a,b]$，分割成$2^{l}-1$等分；一共$2^l$个分点，从小到大，用二进制数表示。精度为$\frac{b-a}{2^l-1}$。误差总是存在的，也可以把区间$[a,b]$，分割成$2^{l}$等分，但不去编码最后一个分点。无限区间，可以表示成一个整数和一个单位长度的区间的组合。

**例**

有些多元函数优化问题，可以直接把多元数组作为个体。此时，个体就是自己的编码。在背包问题中，每个变量都是0或1，因此是一个天然的二进制编码。

每个个体都有一个非常重要的属性，适应度。

**定义（适应度函数）**

适应度函数是用来计算个体适应度的函数，是一个$H\to \R$函数。当$\R$被替换成某个偏序集时，我们可以称之为抽象适应度函数。



适应度函数通常就是优化问题的目标函数，即个体的目标函数值就是个体编码的适应度。两者也可以不同，但最优解应该是相同的。适应度函数可以对目标函数加以改造后获得。

- 绝对变换
  $$
  F(x)=\sigma(f(x))
  $$
  其中$\sigma$一般是有界单调函数，如$\tanh$函数。因为$f$通常是有界的，所以也可选用$\sigma(x)=e^x$。这类适应度函数只考虑目标函数。

- 相对变换

  相对变换可以在绝对变换的基础上定义。它依赖种群的信息。下面这种归一化处理，可以直接作为个体的繁殖概率。

$$
J(x,P)=\frac{F(x)}{\sum_{x\in P}F(x)}
$$

- 自适应变换

  作为相对变换的推广，它充分利用种群的统计信息。下面这种线性变换是为了保证在不该变均值的情况下，适当拉升最大值，即$F_a=f_a,F_\max=cf_\max,1<c\leq 2$。
  $$
  F(x)=\alpha f(x)+\beta,\\
  \alpha=\frac{c-1}{f_{\max}-f_{a}}f_\max,\alpha=\frac{f_\max-cf_{a}}{f_{\max}-f_{a}}f_a
  $$

- 智能化变换

  这类变换会考虑更多的信息，如种群代数，受其他算法的启发。模拟退火遗传算法采用
  $$
  J(x)=\frac{F(x)}{\sum_{x\in P}F(x)},F(x)=e^{f(x)/T},
  $$
  其中$T=c^{n-1}T_0,0<c<1$(可选$c=0.99$)，$n$是种群代数。该适应度函数也可以作为个体的选择概率。



#### 演化算子

所有演化算子都可以概括为$T:H^m\to F(H^n)$，其中$F(H^n)$表示$H^n$上的分布。

##### 选择算子

设$t$代种群$X$，则$t+1$代种群中的个体按照下述概率生成
$$
P(X'_j=X_i)=p(X_i)n(X_i)
$$
一般地，
$$
P(X'\in A)=\sum_{X_i\in A}p(X_i)n(X_i)
$$
(1) 给出了选择算子的转移概率$P(S(X)|X)$。一般规定$p(x)\sim f(x)$（省略了归一化）， $f$可能是目标函数，也可能是改造过的适应度函数。选择时，我们还可以进一步改造，比如（适应度过小的个体会被直接淘汰）


$$
p(x)\sim \begin{cases}
0,f(x)<\epsilon,\\
f(x),o.w.
\end{cases}
$$



通常选择算子总是优先选取最优个体，这类选择叫排序型选择或单调选择。另一类是非单调选择，如Disruptive选择$p(x)\sim u(x), u(x)=|f(x)-\bar{f}|$, $\bar{f}$是种群平均适应度。这种方案会优先选择较好和较坏的个体。



##### 变异算子

直觉告诉我们，一个染色体通过若干次变异都有可能变成任意一个染色体。这个事实使变异算子能使个体遍历整个可行域。以0-1编码为例，如果单个基因的变异概率是$p_m$，则我们有
$$
P(M(X)=Y)=(1-p_m)^{L-d}p_m^d,
$$
其中$d$是$X,Y$之间的Hamming距离。特别地，单个基因变异操作为
$$
P(M(x)=y)=\begin{cases}
1-p_m,&x=y,\\
p_m,&x\neq y.
\end{cases}
$$

显然变异算子可以搜索整个个体空间。

##### 杂交算子

杂交算子并不是必需的，在某些GA变体中，它被省去了。杂交算子并不能在基因水平上发生变化，可能无法遍历整个可行域。如果每个个体的某一位都一样，那么这个它们的杂交后代也是如此。杂交算子可以搜索的范围是包含种群的最小"子空间"。这种子空间是设定全空间中个体某几位为常数获得的，即模式。



杂交的过程一般表述为：从种群中选择两个个体作为母体，交换它们的部分基因，产生两个新的个体。



杂交算子可分为单点杂交和多点杂交，也可分为定点和不定点。

我们来介绍比较常用的定点单点杂交。

**定点的单点杂交操作**
$$
C(a_1\cdots a_l,b_1\cdots b_l, i)=a_1\cdots a_i b_{i+1}\cdots b_l,
$$
其中$i$是杂交点。所谓不定点，就是说$i$是一个随机变量。

1. 单点杂交算子,

2. $$
   P\{C(x,y)=z\}=\begin{cases}\frac{ap}{l}, &z\neq x,\\ (1-p)+\frac{ap}{l},&z=x,\end{cases}
   $$

   其中$a(i,j,k)$是$i,j$杂交生成$k$的基因位数。



人类单个染色体的杂交是单点的。

#### 模式与种群增长方程

模式的概念在分析GA的搜索能力和收敛性时都很有用，甚至也被应用于算法实现。

**定义**

个体空间中的超平面称为个体模式或染色体模式。一般形式为
$$
L=\{x_1x_2\cdots x_l\in H|x_{i_k}=a_{i_k},k=1,2,\cdots,K\}
$$
设$\Gamma=\{0,1\}，l=4$, 则个体可能是$0100$或者$1000$。$*0*0=\{0000,1000,0010,1010\}$就是一种个体模式，其中$*$是通配符，表示该位可以是$0$或$1$。个体模式中的确定位个数是该模式的阶数，记为$o(L)=K$，如$o(*0*0)=2$。第一个确定位到最后一个确定位的距离是该模式的定义长度，记为$\delta(L)=i_K-i_1$，如$o(*0*0)=1$。

个体集$X$的平均适应度$f(X)=\frac{\sum_{x\in X}f(x)}{|X|}$。

**种群增长方程**

设$P(t)$是GA第$t$代种群，$L$是任意模式，种群中满足模式的个体数$N_L(t)=|P(t)\cap L|$。
$$
N_L(t+1) \geq N_L(t)(1-p_c\frac{\delta(L)}{L-1})(1-p_mo(L))\frac{f_L(t)}{f(t)},
$$
其中$f(t)$是$P(t)$的平均适应度，$f_L(t)$是$P(t)\cap L$适应度。近似形式为
$$
N_L(t+1) \geq N_L(t)\frac{f_L(t)}{f(t)},
$$
因为$p_c\frac{\delta(L)}{L-1},p_mo(L)$通常会很小。当$\frac{f_L(t)}{f(t)}>1+C$时，即$f_L$始终高出$f$某个常数倍，我们得到了生物学中常见的指数增长方程
$$
N_L(t) \geq N_L(0)(1+C)^t.
$$


这个结论可被表述为：具有低阶、短定义长度，且平均适应度高于种群适应度的模式以指数级增长。

### 算法框架

#### 基本框架

基本框架表明，GA本质上是一个齐次Markov链。

#### 简单实例



#### 混合算法

### 理论分析简介

#### Markov 链分析

#### 公理化分析

#### 鞅分析

## DEAP 入门

本章开始介绍遗传算法框架DEAP（Distributed Evolutionary Algorithms in Python）。如官网https://deap.readthedocs.io/en/master/index.html上所说，DEAP是一种新的进化计算框架，用于快速原型和思想测试。

### 简单的例子

先来"解剖"一个例子：解一个玩具型优化问题，显然最优解$x=a$。
$$
\min_{x\in\R^n} \|x-a\|
$$

通过这个例子，读者就可以自行设计遗传算法，解决实际问题，同时也可以了解DEAP的设计风格。代码中的注释基大致解释了整个程序的思想。

```python
# 导入相关模块
import numpy as np
import numpy.linalg as LA
from deap import base, creator, tools, algorithms

a = np.random.random(IND_SIZE)
def evaluate(individual):
    """
    计算个体适应度
    
    参数
        individual: {Individual} -- 代表个体的数组
    返回
        元组，与weights匹配
    """
    return LA.norm(individual-a),

# 定义适应度类，作为base.Fitness子类，包含weights属性
creator.create("FitnessMin", base.Fitness, weights=(-1,))
# 定义个体类，作为np.ndarray子类，包含fitness属性
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

"""定义工具箱：
1. 注册构造个体、总群的方法
2. 定义适应度函数
3. 定义并注册遗传操作函数
"""

IND_SIZE = a.shape[0]
toolbox = base.Toolbox()
toolbox.register("gene", np.random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.gene, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 采用DEAP提供的遗传操作函数：交叉、变异、选择
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 创建种群，运行遗传算法
pop = toolbox.population(n=50)
algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.5, mutpb=0.1, ngen=1000, verbose=False)
ind = tools.selBest(pop, 1)[0]
print(f'最优个体(适应度): {ind} ({ind.fitness.values[0]})')
print(f'实际最优解: {a}')

""" 输出
最优个体(适应度): [0.77623954 0.83563347 0.40623318 0.00217375 0.64287153 0.09483329
 0.40491052 0.96871637 0.36552846 0.78586878] (0.002937675111359941)
实际最优解: [0.77615001 0.83562004 0.40533762 0.00194036 0.64120192 0.09558217
 0.40504672 0.96662974 0.36563741 0.78568884]
 """
```

这就是一个多元函数的最小值计算通用程序。读者唯一要改的，是`evaluate`函数，它应该是你的目标函数，其输入是一个`numpy.array` (Individual 是`numpy.ndarray`的子类，操作上与`numpy.array`无异)。例如，把适应度函数改成，

```python
b = np.random.random(m)
A = np.random.random((m, IND_SIZE))
def evaluate(individual):
    return LA.norm(A @ individual-b),
```

就可以解方程组$Ax=b$ (最小二乘解).

**注** 书中代码是为了清楚展示程序内容，不便于过多封装，不影响阅读的代码也省略了。但在上传的代码中，我们实现了封装，可以直接使用。

**注** 上述例子参考https://deap.readthedocs.io/en/master/overview.html。官网上还有许多例子https://deap.readthedocs.io/en/master/examples/index.html。

**注** 特意把交叉算子独立出来写，运行上述程序前，应先加入这段代码。上传的源码会把包括`cxTwoPointCopy`在内的辅助性函数都保存在`utils.py`文件里，因此可用`from utils import *`导入。参考https://deap.readthedocs.io/en/master/examples/ga_onemax_numpy.html。

```python
def cxTwoPointCopy(ind1, ind2):
    """本书一般用numpy.array表示个体，需要避免下述情况
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5,6,7,8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2
```



简单讲一下DEAP的设计模式。它应该采用了工厂模式。



遗传算法通常要解决一个编码问题。最常见的是把实数用二进制数组表示。实现这种转换，主要考虑小数位数。二进制是遗传算法最合适的数值表示。至少，二进制表示非常适合收敛性方面的理论分析。

### 简单应用

遗传算法是通用解法，其应用数不甚数。本节讲几个有实际意义的应用，有些有明显的实践背景、有些则有理论意义。通过这些应用，读者可以了解 DEAP 的 API的功能和用法，不必去阅读文档和源码，从中选择合适的方法去解决自己遇到的问题。

TSP采用0-1编码，是不错的简单实例，原本可以作为第一个例子，但是这个例子被使用的太频繁。关于0-1编码，本文选用了有点新意关联规则实例。

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
toolbox.register("mate", cxTwoPointCopy)
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

'''输出：
最优个体: [0 1 0 1 1 1 1 0] 
是否满足条件: 是
总价值: 185.0
'''
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

#### 关联规则

另一个可以直接用0-1编码的问题是关联规则的挖掘。一个关联规则的著名模型，购物模型是这样表述的：设有$N$个购物单，$p$种商品，$x_{ij}=1$若第$i$个购物单包含商品$j$，否则$x_{ij}=0$，目标函数（支持度）为
$$
T(J)=P(X_j=1,j\in J)\sim\frac{1}{N}\sum_{i=1}^N\prod_{j\in J}x_{ij}, J\subset\{1,2,\cdots, p\}.
$$
另一个可能更有意义的目标函数（置信度）是
$$
C(A\Rightarrow B)=P(X_j=1,j\in B|X_j=1, j\in A)=\frac{T(A\cup B)}{T(A)}, A\cap B=\emptyset.
$$
置信度的意思是用商品集$A$推断同时买商品集$B$的概率。Apriori 算法被用来解决这类问题，其任务是找出所有满足$T(A\Rightarrow B)=T(A\cup B)>t, C(A\Rightarrow B)>c$的$A\Rightarrow B$，而我们直接最大化目标函数$C$， 约束条件为$T>t$。

每个遗传算法的个体应该可以唯一标示$(A,B)$。我们选用$2p$长的0-1向量表示，前$p$位表示$A$，后$p$位表示$B$。

```python
# 导入相关模块

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


'''输出：
sex,cp,trestbps,chol,thalach,thal => age,cp,chol,thalach,thal # Confidence:1.0  Support:0.33993399339933994
'''
```

把适应度函数改成如下函数，可以获得更多的可能结果。一般来说，GA并不能像 Apriori 算法那样找出所有满足条件的规则，但适当规模的种群还是能演化出令人满意的结果的。

```python
t = 0.2
s = 0.8

def evaluate(individual):
    # 返回值: 是否满足条件, 目标函数
    A, B = individual[:p], individual[p:]
    if T(A | B) > t:
        return min((C(A, B),s)), t
    else:
        return 0, T(A|B)
```

正如前文所述，`creator.create`只是一种动态创建类的方法，和`types.new_type`一样。我们也可以用`class`创建个体类，并在类中定义遗传操作方法。还可以定义打印方法，美化输出。

```python
# 导入相关模块

# 描述问题
df = pd.read_csv('heart.csv')
df = df >0
N, p = df.shape

# 定义 T 与 C

# 设置参数，构造关联规则类
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
        self.A, other.A = tools.cxTwoPoint(self.A, other.A)
        self.B, other.B = tools.cxTwoPoint(self.B, other.B)
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


# 构造工具箱
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

def unique(pop):
    popu = [pop[0]]
    print('popu', popu)
    for ind in pop[1:]:
        for ind1 in popu:
            if ind == ind1:
                break
        else:
            popu.append(ind)
    return popu

best = unique(tools.selBest(pop, None))
for ind in best:
    if ind.support > t:
        print(ind, ind.confidence, ind.support)

​```输出
age,oldpeak => trestbps,chol,thalach 1.0 0.6732673267326733
age,slope,ca => trestbps,chol,thalach 1.0 0.40264026402640263
​```
```

很多规则其实是多余的，比如在保证$C(A\Rightarrow B)=C(A'\Rightarrow B')$的情况下，$A\subset A', B'\subset B$，则后者是多余的。下面的函数可以去掉多余的规则。

```python
def clean(pop):
    popc = [pop[0]]
    for ind in pop[1:]:
        flag = True
        for ind1 in popc:
            if ind < ind1:
                popc.remove(ind1)
            elif ind > ind1 or ind == ind1:
                flag = False
                break
        if flag:
            popc.append(ind)
    return popc
  
for ind in clean(best):
    if ind.support > t:
        print(ind, ind.confidence, ind.support)
```



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
toolbox.register("mate", tools.cxTwoPointCopy)
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
# 导入相关模块

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
        toolbox.register("mate", cxTwoPointCopy)
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

运行完这个例子，会让人感受到标准的GA，不仅运行效率低而且得不到很好的结果，不能与专业算法相比。上一章已经提到遗传算法的改进，一种是考虑专业领域的知识，另一种是在不考虑专业知识的前提下，提高算法的自适应能力。

#### 神经网络结构优化



#### 多元时间序列预测



#### 参数选择



### 算法细节改进

`algorithms.eaSimple`是遗传算法最常规的实现。它大致的流程是：


- 计算初始种群中个体适应度
- 对每一代循环:
    - 选择下一代总群
    - 遗传操作
    
      `offspring = varAnd(offspring, toolbox, cxpb, mutpb)`
    - 计算个体适应度并更新总群

这里的核心就是`varAnd`函数，整个算法就是不断执行该函数。读者要设计自己的遗传算法，可以重新实现这个函数。

为了便于理解，精简代码如下。

```python
def varAnd(population, toolbox, cxpb, mutpb):

    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, ngen):

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the current population by the offspring
        population[:] = offspring

    return population
```

**注** DEAP作者，考虑到个体的适应度可能被重复计算，因此将适应度存在属性`fitness`里，只有新生成的个体，才会被计算适应度。`individual`是可变变量，会被拷贝一份，修改其中的染色体编码模拟遗传操作，同时删除适应度，这样的个体因为没有了适应度被放入列表`invalid_ind`，只有这些个体被重新计算适应度。

### 算法可视化




## DEAP 进阶

### 多线程实现

### 扩展

## DEAP 复杂应用

### 融合智能局部搜索算法

这或许是GA最有意义的话题。如果遗传算法是受Darwin进化论的启发，那么局部搜索的GA是受Lamarck理论的启发。

## DEAP 源码解读



## 其他遗传算法框架

### 其他框架介绍

### 自制框架


