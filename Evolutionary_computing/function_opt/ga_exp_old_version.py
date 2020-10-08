from deap import tools, base
from multiprocessing import Pool
from ga_scheme import eaMuPlusLambda
from deap.algorithms import eaMuPlusLambda
from numpy import random as rnd
import numpy as np
from deap import creator
from deap import benchmarks
import pandas as pd

creator.create("BaseFitness", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.BaseFitness)

# Nonuniform creep mutation
def mutation(individual):
    n = len(individual)
    for i in range(n):
        if rnd.random() < n * 0.15:
            individual[i] += rnd.normal(0.0, 0.2)
            individual[i] = np.clip(individual[i], -5, 5)
    return individual,


class SimpleGAExperiment:
    def factory(self):
        return rnd.random(self.dimension) * 10 - 5

    def __init__(self, function, dimension, pop_size, iterations, mutation_prob, crossover_prob):
        self.pop_size = pop_size
        self.iterations = iterations
        self.mut_prob = mutation_prob
        self.cross_prob = crossover_prob

        self.function = function
        self.dimension = dimension

        # self.pool = Pool(5)
        self.engine = base.Toolbox()
        # self.engine.register("map", self.pool.map)
        self.engine.register("map", map)
        self.engine.register("individual", tools.initIterate, creator.Individual, self.factory)
        self.engine.register("population", tools.initRepeat, list, self.engine.individual, self.pop_size)
        self.engine.register("mate", tools.cxOnePoint)
        #self.engine.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
        self.engine.register("mutate", mutation)
        self.engine.register("select", tools.selTournament, tournsize=4)
        # self.engine.register("select", tools.selRoulette)
        self.engine.register("evaluate", self.function)

    def run(self):
        pop = self.engine.population()
        hof = tools.HallOfFame(3, np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = eaMuPlusLambda(pop, self.engine, mu=self.pop_size, lambda_=int(self.pop_size*0.8), cxpb=self.cross_prob, mutpb=self.mut_prob,
                                  ngen=self.iterations,
                                  stats=stats, halloffame=hof, verbose=True)
        print("Best = {}".format(hof[0]))
        print("Best fit = {}".format(hof[0].fitness.values[0]))
        return log, {'best': hof[0], 'best_fit': hof[0].fitness.values[0]}

single = True
from functions import rastrigin
if __name__ == "__main__":

    def function(x):
        res = rastrigin(x)
        return res,

    if single == True:
        # Single launch
        dimension = 100
        pop_size = 200
        iterations = 1000
        mutation_prob = 0.6
        crossover_prob = 0.3
        scenario = SimpleGAExperiment(function, dimension, pop_size, iterations, mutation_prob, crossover_prob)
        log, final_values = scenario.run()
        from draw_log import draw_log
        draw_log(log)
    else:
        # Multiple launches
        dimension = 100
        iterations = 150
        pop_sizes = [100, 200, 500]
        mutation_probs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        crossover_probs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

        # Full grid search
        data = []
        for pop_size in pop_sizes:
            for mutation_prob in mutation_probs:
                for crossover_prob in crossover_probs:

                    try:
                        # Для каждой конфигурации запускаем алгоритм 10 раз
                        best_values = []
                        for launch in range(0,10):
                            scenario = SimpleGAExperiment(function, dimension, pop_size, iterations, mutation_prob, crossover_prob)
                            log, final_values = scenario.run()
                            best_values.append(final_values.get('best_fit'))

                        # Усредняем значения фитнесс функции
                        best_values = np.array(best_values)
                        mean_value = np.mean(best_values)
                        data.append([pop_size, mutation_prob, crossover_prob, mean_value])
                    except Exception:
                        pass

        dataframe = pd.DataFrame(data, columns = ['Population size', 'Mutation', 'Crossover', 'Scores'])
        dataframe.to_csv('D:/ITMO/Simple_EA_scores.csv', index=False)
