from population import AutomataPopulation as Population
from settings.settings import *

if __name__ == '__main__':
    population = Population(POPULATION_SIZE)
    population.train(GENERATIONS)