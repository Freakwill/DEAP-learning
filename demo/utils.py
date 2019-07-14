#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from deap import base, creator, tools

def default_tool(gene_length, ind_size, evaluate, creator):

    toolbox = base.Toolbox()
    toolbox.register("gene", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.gene, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
