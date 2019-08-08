#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from deap import base, creator, tools, algorithms

from utils import cxTwoPointCopy, cxTwoPoint2DArray

from sklearn.neural_network import MLPClassifier

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/Users/william/Folders/Database/mnist/', one_hot=True)

@np.vectorize
def ReLU(x):
    return max([x,0])

@np.vectorize
def u(x):
    return x>0


class ANN(object):
    '''
    m: Input size
    n: Output size
    '''
    def __init__(self, m, n, h=8):
        self.m = m
        self.n = n
        self.h = h
        self.layer1 = np.random.random((m, h))
        self.bias1 = np.random.random(h)
        self.layer2 = np.random.randint(0, 2, (h,n))
        self.bias2 = np.random.random(n)

    def predict_prob(self, X):
        Z = np.dot(X, self.layer1) + np.tile(self.bias1, (X.shape[0], 1))
        Z = ReLU(Z)
        Y = np.dot(Z, self.layer2) + np.tile(self.bias2, (X.shape[0], 1))
        Y = softmax(Y, axis=1)
        return Y

    def predict(self, X):
        Y = self.predict_prob(X)
        return [np.argmax(y) for y in Y]


    def evaluate(self, X, T):
        Y = self.predict_prob(X)
        S = np.sum([entropy(t, y) for y, t in zip(Y, T)])
        return S,

    def accuracy(self, X, T):
        N = X.shape[0]
        Y = self.predict(X)
        k = 0
        for y, t in zip(Y, T):
            if t[y] == 1:
                k+= 1
        return k / N


    def mate(self, other):
        self.layer1, other.layer1 = cxTwoPoint2DArray(other.layer1, self.layer1)
        self.bias1, other.bias1 = cxTwoPointCopy(self.bias1, other.bias1)
        self.layer2, other.layer2 = cxTwoPoint2DArray(self.layer2, other.layer2)
        self.bias2, other.bias2 = cxTwoPointCopy(self.bias2, other.bias2)

        return self, other

    def mutate(self, mu=0, sigma=0.05, indpb=0.05):
        r, c = self.layer1.shape
        for i in range(r):
            for j in range(c):
                if np.random.random() < indpb:
                    self.layer1[i,j] += np.random.rand() * sigma
        self.bias1, = tools.mutGaussian(self.bias1, mu, sigma, indpb)

        r, c = self.layer2.shape
        for i in range(r):
            for j in range(c):
                if np.random.random() < indpb:
                    self.layer2[i,j] = 1 - self.layer2[i,j]
        self.bias2, = tools.mutGaussian(self.bias2, mu, sigma, indpb)
        return self,

class GAANN(MLPClassifier):
    """GA for ANN
    """

    def config(self, X, Y):
        # configuration for GA
        
        creator.create("FitnessMin", base.Fitness, weights=(-1,))
        creator.create("Individual", ANN, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        # import multiprocessing
        # pool = multiprocessing.Pool()
        # toolbox.register("map", pool.map)
        toolbox.register("individual", creator.Individual, X.shape[1], Y.shape[1])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", ANN.mate)
        toolbox.register("mutate", ANN.mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", lambda ind:ind.evaluate(X,Y))
        
        def ga(pop=None):
            if pop is None:
                pop = toolbox.population(n=200)
            pop, logbook = algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.6, mutpb=0.17, ngen=100, verbose=False)
            pop, logbook = algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.5, mutpb=0.1, ngen=300, verbose=False)
            pop, logbook = algorithms.eaSimple(pop, toolbox=toolbox, cxpb=0.4, mutpb=0.07, ngen=100, verbose=False)
            return pop, logbook
        return ga

    def fit(self, X, Y, *args, **kwargs):
        pop, self.logbook = self.config(X,Y)()
        self.best = tools.selBest(pop,1)[0]
        return self

    def predict(self, X):
        return self.best.predict(X)

    def evaluate(self, X, Y):
        return self.best.evaluate(X, Y)

    def accuracy(self, X, Y):
        return self.best.accuracy(X, Y)

X = mnist.train.images[:200, :]
Y = mnist.train.labels[:200, :]

gaann = GAANN()
gaann.fit(X, Y)
print(gaann.predict(X[:1,:]), Y[:1,:])
print(gaann.accuracy(X, Y))
