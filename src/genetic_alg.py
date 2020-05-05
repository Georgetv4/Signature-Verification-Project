"""
This file describes Genetic Algorithm steps
"""
from src.config import EPOCH
import random as rm


class GeneticAlg:
    def __init__(self):
        self.y = lambda x: (x * x + 1) / 12 - 44
        self.init_population = [1, 2, 5, 6, 7]
        self.ev = list(map(self.y, self.init_population))
        print(self.ev)

    def crossover(self):
        pass

    def mutation(self, x):
        return x + rm.randint(-5, 5)

    def selection(self, population, pre_population):
        ev = list(map(self.y, population))
        pre_ev = list(map(self.y, pre_population))

        cort = []
        pre_cort = []
        for i in range(len(population)):
            cort.append([population[i], ev[i]])
            pre_cort.append([pre_population[i], pre_ev[i]])

        by_y = lambda e: e[1]
        sorted(cort, key=by_y)
        sorted(pre_cort, key=by_y)

        new_population = [cort[0][0], cort[1][0], pre_cort[0][0], pre_cort[1][0], rm.randint(-10, 10)]
        return new_population

    def evolve(self, epoch=EPOCH):
        population = self.init_population.copy()

        for i in range(epoch):
            pre_population = population
            population = list(map(self.mutation, population))
            population = self.selection(population, pre_population)

        ev = list(map(self.y, population))
        ev.sort()
        print(ev[0])


GeneticAlg().evolve()
