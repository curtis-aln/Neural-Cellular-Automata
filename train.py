from population import AutomataPopulation as Population

POPULATION_SIZE = 100
GENERATIONS = 100

if __name__ == '__main__':
    population = Population(POPULATION_SIZE)
    population.train(GENERATIONS)