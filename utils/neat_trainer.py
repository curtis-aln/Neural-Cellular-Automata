# General Class used for training NEAT networks using the neat algorithm.

import neat
import pickle

from neat.nn import FeedForwardNetwork as Network
from neat.genome import DefaultGenome as Genome
from neat.population import Population


class NeatFileManager:
    def __init__(self, file_config_path = "", population_save_file = "", best_genome_save_file = "") -> None:
        # the path to the config file containing the information about the NEAT algorithm.
        self.file_config_path = file_config_path
        
        # contains the whole population
        self.population_save_file = population_save_file

        # only contains the best genome of the population
        self.best_genome_save_file = best_genome_save_file
    

    def get_config(self, file_config_path : str):
        return neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation, file_config_path)
    

    def read_file(self, file_path : str):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    

    def get_best_genome_from_file(self, filename : str = "") -> Genome:
        return self.read_file(filename if filename != "" else self.best_genome_save_file)
        
    
    def get_population_from_file(self, filename : str = "") -> Population:
        return self.read_file(filename if filename != "" else self.population_save_file)


    def save_data(self) -> None:
        if self.best_genome_save_file != "":
            with open(self.best_genome_save_file, 'wb') as f:
                pickle.dump(self.population.best_genome, f)
        
        if self.population_save_file != "":
            with open(self.population_save_file, 'wb') as f:
                pickle.dump(self.population, f)
    
    def genome_to_network(self, genome : Genome) -> Network:
        return Network.create(genome, self.config)



class NeatTrainer(NeatFileManager):
    def __init__(self, eval_genomes_func = None, config_file_path="", population_save_file="", best_genome_save_file="") -> None:
        super().__init__(config_file_path, population_save_file, best_genome_save_file)

        if config_file_path != "":
            self.config = self.get_config(config_file_path)

        # the function used to determine the fitness of the agents in the population. one generation
        self.eval_genomes_func = eval_genomes_func

        self.population = None if population_save_file == "" else self.read_file(population_save_file)
    

    def set_population(self, population : Population) -> None:
        self.population = population
    

    def get_generation(self) -> int:         return self.population.generation
    def get_population(self) -> Population:  return self.population
    def get_best_genome(self) -> Genome:     return self.population.best_genome

    
    def make_network_from_genome(self, genome : Genome) -> Network:
        return Network.create(genome, self.config)


    def train_population(self, generations : int) -> tuple[Network, Genome]:
        # if there is no current population loaded start a new one
        if self.population is None:
            self.population = neat.Population(self.config)
            self.population.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            self.population.add_reporter(stats)
        
        # run the training
        best_genome = self.population.run(self.eval_genomes_func, n=generations)
        self.best_network = self.make_network_from_genome(best_genome)

        return (self.best_network, self.population.best_genome)