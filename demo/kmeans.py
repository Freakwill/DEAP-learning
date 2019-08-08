#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import pairwise_distances
from deap import base, creator, tools, algorithms

from sklearn.cluster import KMeans


class GAKMeans(KMeans):
    """GA for KMeans
    
    the individual of GA is the sequence of labels of samples
    the best one in last generation is the result of clustering.
    
    Extends:
        KMeans
    """
    def __init__(self, size=50, ngen=100, local_search = False, *args, **kwargs):
        super(GAKMeans, self).__init__(*args, **kwargs)
        self.n_clusters=kwargs['n_clusters']
        self.local_search = local_search
        self.size = size
        self.ngen = ngen

    def config(self, X):
        # configuration for GA
        K = self.n_clusters
        # D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))
        def evaluate(individual):
            W = 0
            for k in range(K):
                Ck = [g==k for g in individual]
                # Dk = scipy.spatial.distance.squareform(D[ck, :][:, ck])
                Xk = X[Ck, :]
                ck = Xk.mean(axis=0)
                Dk = pairwise_distances(Xk, [ck])**2
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
            pop = toolbox.population(n=self.size)
            pop, _= algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.55, mutpb=0.1, ngen=self.ngen, verbose=False)
            return pop, _
        return ga

    def fit(self, X, *args, **kwargs):
        super(GAKMeans, self).fit(X, *args, **kwargs)

        pop, _ = self.config(X)()
        best = tools.selBest(pop,1)[0]
        self.labels_ = np.array(best)
        self.cluster_centers_= np.vstack([X[[g==k for g in self.labels_], :].mean(axis=0) for k in range(self.n_clusters)])
        self.inertia_ = best.fitness.values[0]
        return self
