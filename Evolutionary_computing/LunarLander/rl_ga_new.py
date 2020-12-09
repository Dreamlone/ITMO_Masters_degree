from deap import tools, base, creator
import numpy as np
import numpy.random as rnd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import gym
import random
from copy import deepcopy
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

def varOr(population, toolbox, lambda_, cxpb, mutpb):
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            del ind2.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(toolbox.clone(random.choice(population)))

    return offspring


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        if halloffame is not None:
            for ind in halloffame:
                population.append(toolbox.clone(ind))
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def read_log(log):
    avg_list = list()
    std_list = list()
    min_list = list()
    max_list = list()
    gen_list = list()
    for g in log:
        avg_list.append(g['avg'])
        std_list.append(g['std'])
        min_list.append(g['min'])
        max_list.append(g['max'])
        gen_list.append(g['gen'])
    return np.array(gen_list), np.array(avg_list), np.array(std_list), np.array(max_list), np.array(min_list)

def draw_log(log):
    gen_list, avg_list, std_list, max_list, min_list = read_log(log)
    plt.plot(gen_list, avg_list, label="avg")
    plt.plot(gen_list, min_list, label="min")
    plt.plot(gen_list, max_list, label="max")
    plt.fill_between(gen_list, avg_list-std_list, avg_list+std_list, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

def draw_logs(log1, log2, lab1, lab2):
    gen1, avg1, std1, max1, min1 = read_log(log1)
    gen2, avg2, std2, max2, min2 = read_log(log2)
    plt.plot(gen1, avg1, label=lab1, color="blue")
    plt.plot(gen1, max1, label="{}_max".format(lab1), color="blue", linewidth=2)
    plt.fill_between(gen1, avg1 - std1, avg1 + std1, alpha=0.2, color="blue")
    plt.plot(gen2, avg2, label=lab2, color="orange")
    plt.plot(gen2, max2, label="{}_max".format(lab2), color="orange", linewidth=2)
    plt.fill_between(gen2, avg2 - std2, avg2 + std2, alpha=0.2, color="orange")
    plt.legend()
    plt.tight_layout()
    plt.show()


################################################################################
#                       THE ALGORITHM FOR A HOMEWORK                           #
################################################################################
creator.create("BaseFitness", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.BaseFitness)

class RL_ga_experiment:
    def factory(self):
        individual = list()
        for i in range(len(self.params)):
            if i % 2 == 0:
                individual.append(rnd.normal(0.1, 0.3, size=self.params[i].shape))
            else:
                individual.append(np.zeros(shape=self.params[i].shape))
        return creator.Individual(individual)


    def mutation(self, individual):
        # Individual - 0 [8, 200]
        # Individual - 1 [200, ] [0,0,0,...]
        # Individual - 2 [200, 120]
        # Individual - 3 [120, ] [0,0,0,...]
        # Individual - 4 [120, 4]
        # Individual - 5 [4, ] [0,0,0,...]

        # choice_value = rnd.random()
        # if choice_value < 0.3:
        #     i = 0
        # elif choice_value < 0.6:
        #     i = 2
        # else:
        #     i = 4

        for i in range(len(individual)):
            # Only for 0, 2, 4
            if i == 0:
                if rnd.random() < 0.3:
                    value = rnd.normal(0.0, 5.0)
                else:
                    value = rnd.normal(0.0, 0.05)

                for j in range(len(individual[i])):
                    for k in range(len(individual[i][j])):
                        individual[i][j][k] += value

            elif i == 2:
                if rnd.random() < 0.3:
                    value = rnd.normal(0.0, 10.0)
                else:
                    value = rnd.normal(0.0, 0.05)

                for j in range(len(individual[i])):
                    for k in range(len(individual[i][j])):
                        individual[i][j][k] += value

            elif i == 4:
                if rnd.random() < 0.3:
                    value = rnd.normal(0.0, 10.0)
                else:
                    value = rnd.normal(0.0, 0.05)

                for j in range(len(individual[i])):
                    for k in range(len(individual[i][j])):
                        individual[i][j][k] += value

        return individual,


    def crossover(self, p1, p2):
        c1 = list()
        c2 = list()

        arr_0_to_process = np.asarray([p1[0], p2[0]])
        first = np.mean(arr_0_to_process, axis=0)

        arr_2_to_process = np.asarray([p1[2], p2[2]])
        second = np.mean(arr_2_to_process, axis=0)

        arr_4_to_process = np.asarray([p1[4], p2[4]])
        third = np.mean(arr_4_to_process, axis=0)

        c1.append(first)
        c1.append(deepcopy(p1[1])) # zero
        c1.append(second)
        c1.append(deepcopy(p1[3])) # zero
        c1.append(third)
        c1.append(deepcopy(p1[5])) # zero

        c2.append(deepcopy(p1[0]))
        c2.append(deepcopy(p2[1]))  # zero
        c2.append(deepcopy(p1[2]))
        c2.append(deepcopy(p2[3]))  # zero
        c2.append(deepcopy(p2[4]))
        c2.append(deepcopy(p2[5]))  # zero

        return creator.Individual(c1), creator.Individual(c2)

    def __init__(self, input_dim, l1, l2, output_dim, pop_size, iterations):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l1 = l1
        self.l2 = l2
        self.pop_size = pop_size
        self.iterations = iterations

        self.model = self.build_model()
        self.params = self.model.get_weights()
        self.env = gym.make("LunarLander-v2")

        self.engine = base.Toolbox()
        self.engine.register('map', map)
        self.engine.register("individual", tools.initIterate, creator.Individual, self.factory)
        self.engine.register('population', tools.initRepeat, list, self.engine.individual, self.pop_size)
        self.engine.register('mutate', self.mutation)
        self.engine.register("mate", self.crossover)
        self.engine.register('select', tools.selTournament, tournsize=3)
        self.engine.register('evaluate', self.fitness)

    def compare(self, ind1, ind2):
        result = True
        for i in range(len(ind1)):
            if i % 2 == 0:
                for j in range(len(ind1[i])):
                    for k in range(len(ind1[i][j])):
                        if ind1[i][j][k] != ind2[i][j][k]:
                            return False

        return result

    def run(self):
        pop = self.engine.population()
        hof = tools.HallOfFame(3, similar=self.compare)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register('min', np.min)
        stats.register('max', np.max)
        stats.register('avg', np.mean)
        stats.register('std', np.std)

        pop, log = eaMuPlusLambda(pop, self.engine,
                                  mu=self.pop_size,
                                  lambda_=int(0.8 * self.pop_size),
                                  cxpb=0.2, mutpb=0.8,
                                  ngen=self.iterations,
                                  verbose=True,
                                  halloffame=hof,
                                  stats=stats)
        best = hof[0]
        print("Best fitness = {}".format(best.fitness.values[0]))
        return log, best


    def build_model(self):
        model = Sequential()
        model.add(InputLayer(self.input_dim))
        model.add(Dense(self.l1, activation='relu'))
        model.add(Dense(self.l2, activation='relu'))
        model.add(Dense(self.output_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def fitness(self, individual):
        self.model.set_weights(individual)
        scores = []
        for _ in range(1):
            state = self.env.reset()
            score = 0.0
            for t in range(500):
                self.env.render()
                act_prob = self.model.predict(state.reshape(1, self.input_dim)).squeeze()
                action = rnd.choice(np.arange(self.output_dim), 1, p=act_prob)[0]
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                state = next_state
                if done:
                    break
            scores.append(score)
        return np.mean(scores),

if __name__ == '__main__':
    input_dim = 8
    l1 = 200
    l2 = 120
    output_dim = 4
    pop_size = 10
    iterations = 100

    exp = RL_ga_experiment(input_dim, l1, l2, output_dim, pop_size, iterations)
    log, best = exp.run()

    draw_log(log)

    for _ in range(100):
        exp.fitness(best)
