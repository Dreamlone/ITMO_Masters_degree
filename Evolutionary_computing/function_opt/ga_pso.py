import operator
import random

import numpy
import math

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from draw_log import draw_log, draw_logs
from ga_exp import *
from pso_exp import *
from functions import rastrigin


if __name__ == "__main__":
    def function(x):
        res = rastrigin(x)
        return res,
    dimension = 50
    pop_size = 100
    iterations = 100
    ga = SimpleGAExperiment(function, dimension, pop_size, iterations)
    log = ga.run()

    pso = PSOAlg(pop_size, iterations, dimension, function)
    pop, logbook, best = pso.run()
    draw_logs(log, logbook, "ga", "pso")

