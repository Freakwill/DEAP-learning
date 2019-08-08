#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, random

def transp(body, toolbox, T):
    offspring = toolbox.clone(body)
    offspring = toolbox.get_neighbour(offspring)
    offspring.fitness.values = toolbox.evaluate(offspring)
    D = offspring.fitness.values[0] - body.fitness.values[0]
    if D > 0:
        p = min((1, math.exp(-D/T)))
        if random.random() <= p:
            del offspring.fitness.values
            return offspring
        else:
            return body
    else:
        del offspring.fitness.values
        return offspring


def annealing(body, toolbox, initT, ngen, stats=None,
             verbose=__debug__):
    """Simulated Annealing algorithm
    """
    
    c = 0.99

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the bodys with an invalid fitness
    if not body.fitness.valid:
        body.fitness.values = toolbox.evaluate(body)

    record = stats.compile([body]) if stats else {}
    logbook.record(gen=0, nevals=1, **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        T = initT
        for epoch in range(50):
            offspring = transp(body, toolbox, T)

            # Evaluate the body with an invalid fitness
            if not offspring.fitness.valid:
                offspring.fitness.values = toolbox.evaluate(offspring)

            body = offspring
            T = T * c ** epoch
        # Append the current generation statistics to the logbook
        record = stats.compile([body]) if stats else {}
        logbook.record(gen=gen, nevals=1, **record)
    return body, logbook


if __name__ == '__main__':

    import numpy as np
    import numpy.linalg as LA
    from deap import base, creator, tools, algorithms
    IND_SIZE = 3
    N = 5
    b = np.random.random(N)
    A = np.random.random((N, IND_SIZE))
    def evaluate(body):
        return LA.norm(A @ body-b) / LA.norm(b),

    # 定义适应度类，作为base.Fitness子类，包含weights属性
    creator.create("FitnessMin", base.Fitness, weights=(1,))
    # 定义个体类，作为np.ndarray子类，包含fitness属性
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("gene", np.random.random)
    toolbox.register("body", tools.initRepeat, creator.Individual,
                     toolbox.gene, n=IND_SIZE)

    # 邻解
    toolbox.register("get_neighbour", tools.mutGaussian, mu=0, sigma=0.1, indpb=1)

    def deco(f, *args, **kwargs):
        def ff(*args, **kwargs):
            return f(*args, **kwargs)[0]
        return ff

    toolbox.decorate("get_neighbour", deco)
    toolbox.register("evaluate", evaluate)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("value", lambda x: x[0])

    # 创建种群，运行遗传算法
    ind = toolbox.body()
    ind, logbook = annealing(ind, toolbox=toolbox, initT=1, ngen=100, stats=stats, verbose=False)

    print(f'最优个体(适应度): {ind} ({ind.fitness.values[0]})')
    x= LA.lstsq(A, b, rcond=None)
    print(f'实际最优解(适应度): {x[0]} ({x[1]})')

    gen, value = logbook.select("gen", "value")

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, value, "b-", label="Value")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")
    plt.show()



