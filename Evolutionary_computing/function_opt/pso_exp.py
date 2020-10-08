import operator
import random

import numpy as np
import math
import numpy.random as rnd

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from draw_log import draw_log

creator.create("BaseFitness", base.Fitness, weights=(1.0,))
creator.create("Particle", np.ndarray, fitness=creator.BaseFitness, speed=None, smin=None, smax=None, best=None)


class PSOAlg:
    def generate(self):
        return rnd.uniform(self.pmin, self.pmax, self.dimension)

    def particle_generation(self):
        particle = tools.initIterate(creator.Particle, self.generate)
        particle.speed = rnd.uniform(self.smin, self.smax, self.dimension)
        particle.smin = self.smin
        particle.smax = self.smax
        return particle

    def updateParticle(self, part, best):
        v1 = (part.best - part) * rnd.uniform(0, self.c1)
        v2 = (best - part) * rnd.uniform(0, self.c2)
        part.speed = np.clip(part.speed * self.w + v1 + v2, self.smin, self.smax)
        part[:] = np.clip(part[:] + part.speed, self.pmin, self.pmax)

    def __init__(self, pop_size, iterations, dimension, function):
        self.pop_size = pop_size
        self.iterations = iterations
        self.dimension = dimension
        self.function = function
        self.c1 = 0.8
        self.c2 = 0.7
        self.w = 1.0
        self.pmin = -5.0
        self.pmax = 5.0
        self.smin = -2.0
        self.smax = 2.0

        self.toolbox = base.Toolbox()
        self.toolbox.register("particle", self.particle_generation)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)
        self.toolbox.register("update", self.updateParticle)
        self.toolbox.register("evaluate", self.function)

    def run(self):
        pop = self.toolbox.population(n=self.pop_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        GEN = self.iterations
        best = None

        for g in range(GEN):
            for part in pop:
                part.fitness.values = self.toolbox.evaluate(part)
                if part.best is None or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if best is None or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
            for part in pop:
                self.toolbox.update(part, best)

            # Gather all the fitnesses in one list and print the stats
            logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
            print(logbook.stream)

        return pop, logbook, best


from functions import rastrigin

if __name__ == "__main__":
    def function(x):
        res = rastrigin(x)
        return res,


    dimension = 50
    pop_size = 100
    iterations = 1000
    pso_exp = PSOAlg(pop_size, iterations, dimension, function)
    pop, logbook, best = pso_exp.run()
    print("best")
    print(best)
    print("speed")
    print(pop[0].speed)
    draw_log(logbook)
