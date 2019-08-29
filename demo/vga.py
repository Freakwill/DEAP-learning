#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from deap import tools


k1, k2, k3, k4 = 0.9, 0.3, 0.1, 0.09
epsilon = 0.00001

# def llt(f, favg, fmax, k1, k2):
#     # Limited Linear transform
#     if f < favg:
#         return k1
#     else:
#         return k1 - k2*(f-favg+epsilon)/(fmax-favg+epsilon)


def adaptive_varAnd(population, toolbox):

    fitnesses = np.array([ind.fitness.wvalues[0] for ind in population])
    fitnesses /= fitnesses.mean()
    fmax = fitnesses.max()

    offspring = [toolbox.clone(ind) for ind in population]
    # Apply crossover and mutation on the offspring
    k = fmax-1
    for i, ff in enumerate(zip(fitnesses[::2], fitnesses[1::2])):
        f = max(ff)
        if f<=1 or k<epsilon:
            cxpb = k1
        else:
            cxpb = k1 - k2*(f-1)/ k
        i *= 2
        if random.random() < cxpb:
            offspring[i], offspring[i+1] = toolbox.mate(offspring[i], offspring[i+1])
            del offspring[i].fitness.values, offspring[i+1].fitness.values

    for i, f in enumerate(fitnesses):
        # mutpb = am(f, favg, fmax)
        if f <= 1 or k<epsilon: 
            mutpb = k3
        else:
            mutpb = k3 - k4*(f-1)/k
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring



def eaAdaptive(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = adaptive_varAnd(offspring, toolbox)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook
