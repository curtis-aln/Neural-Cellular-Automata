# a class made to find the correct list of weights which will generate an image after N time steps

import numpy as np
import pygame as pg

from PIL import Image

from automata import CellularAutomata
from scipy.signal import convolve2d
from scipy.ndimage import zoom


from settings.settings import *
from utils.neat_trainer import NeatTrainer, NeatFileManager
from utils.colorama import print_col

from neat.nn import FeedForwardNetwork as Network
from neat.genome import DefaultGenome as Genome

def load_image(filepath : str) -> np.ndarray:
    img = Image.open(filepath).rotate(90, expand=True) # loading image the right way around
    return np.array(img, dtype=np.float32) / 255       # normalized between 0 & 1

def get_desired_and_init_states() -> tuple[np.ndarray]:
    desired_image = load_image(TARGET_IMAGE_PATH)
    image_shape = desired_image.shape
    grid_shape = np.array([image_shape[0], image_shape[1], AUTOMATA_DEPTH])

    # calculating the initial grid state
    init_grid_states = np.zeros(grid_shape)
    idx_a = int(grid_shape[0]/2)
    idx_b = int(grid_shape[1]/2)
    init_grid_states[idx_a][idx_b][0] = 1

    return (desired_image, image_shape, init_grid_states)




# The AutomataPopulation is a class that applies the NEAT algorithm to a population.
class AutomataPopulation(NeatTrainer):
    def __init__(self, population_size : int) -> None:
        # general purpose NEAT class used to train th
        super().__init__(self.eval_genomes, CONFIG_PATH, SAVE_FILE_POPULATION, SAVE_FILE_BEST_GENOME)

        self.desired_image, self.image_shape, self.init_grid_states = get_desired_and_init_states()

        # creating the population
        self.automatas = [CellularAutomata(self.init_grid_states) for _ in range(population_size)]


    def get_best_automata(self):
        best_automata = self.automatas[0]

        for automata in self.automatas:
            if automata.score > best_automata.score:
                best_automata = automata
        
        return best_automata


    # evaluate genomes runs a single generation to get the fitnessess for each neural network.
    def eval_genomes(self, genomes, config):
        # The neat algorithm can decide to increase or decrease the population if seen fit.
        while (len(genomes) > len(self.automatas)):
            self.automatas.append(CellularAutomata(self.init_grid_states))

        # running the generation
        for automata, genome in zip(self.automatas, genomes):
            net = Network.create(genome[1], config)
    
            genome[1].fitness = automata.run(ITERATIONS, net, self.desired_image)

        # logistics, saving, changing targets, and printing data is managed
        gen = self.get_generation()
        if gen % SAVE_FREQ == 0:
            self.save_data()
            print_col("Data Saved", 'green')

        self.save_image()
    

    def save_image(self):
        # getting the final image from the best automata and writing it to a png file
        best_automata = self.get_best_automata()
        final_image = best_automata.get_image()

        # Normalize the pixel values to the range [0, 255]
        normalized_image = (final_image * 255).astype(np.uint8)

        # Desired resolution
        target_height = 1080
        target_width = 1920

        # Compute the zoom factors
        zoom_factor_y = target_height / normalized_image.shape[1]
        zoom_factor_x = target_width / normalized_image.shape[0]

        resized_image = zoom(normalized_image, (zoom_factor_y, zoom_factor_x), order=0)

        # Convert the normalized 2D array to a 3D array if it's grayscale
        if resized_image.ndim == 2:
            resized_image = np.stack((resized_image,) * 3, axis=-1)

        # Create a surface and save it as a PNG file
        surface = pg.surfarray.make_surface(resized_image)
        pg.image.save(surface, f"best/best_automata.png")
    

    def get_best_automata(self) -> CellularAutomata:
        best_automata = self.automatas[0]
        for automata in self.automatas:
            if automata.score > best_automata.score:
                best_automata = automata
        return best_automata
        

    def train(self, generations : int) -> tuple[Network, Genome]:
        print_col("would you like to start training from the saved data? (y/n)", "cyan")
        if input(">>> ") == "y":
            self.set_population_from_file()
        best_network, best_genome = self.train_population(generations)

        print_col("would you like to save the training data? (y/n) ", "cyan")
        if input(">>> ") == "y":
            self.save_data()
        
        return (best_network, best_genome)



class EvolutionManager:
    def __init__(self, render_surface : pg.Surface, train_image_file_location : str) -> None:
        self.render_surface = render_surface

        r = 6
        l = np.zeros((r, r)) + 1
        self.desired_image = convolve2d(self.load_image(train_image_file_location), l, mode='same') / (r*r)
        shape = self.desired_image.shape
        self.grid_shape = np.array([shape[0], shape[1], AUTOMATA_DEPTH])

        self.automatas = self.init_automatas()
        self.calc_grid_diff()
    
    # rendering and user interface
    # todo optimize by using numpy's whole-array operations to normalize the whole array to be between 0 and and 255
    def state_to_color(self, state : float, render_mode = True):
        if (render_mode == False):
            color = max(0, min(1, abs(state))) * 255
            return (color, 0, 0) if state < 0 else (0, color, 0)
        
        color = max(0, min(1, state)) * 255
        return (color, color, color)
    

    def render_desired(self, rect : pg.Rect) -> None: 
        self.render_automata2(self.desired_image, rect)


    def render_current_best_gen(self, rect : pg.Rect) -> None: 
        self.render_automata(self.automatas[0].grid, rect)


    def render_difference_grid(self, rect : pg.Rect) -> None:
        self.render_automata2(self.grid_diff, rect, False)


    def render_previous_best_attempt(self, rect : pg.Rect) -> None: 
        self.render_automata(self.best_grid_saved, rect)


    def render_automata(self, grid_pattern : np.ndarray, rect : pg.Rect, render_mode = True) -> None:
        start_pos = pg.Vector2(rect.x, rect.y)
        size = pg.Vector2(int(rect.w / grid_pattern.shape[0]), int(rect.h / grid_pattern.shape[1]))
        transposed = np.transpose(grid_pattern, (2, 0, 1))

        for x in range(len(grid_pattern)):
            for y in range(len(grid_pattern[x])):
                color = self.state_to_color(transposed[ANALYZE_LAYER][x][y], render_mode)
                rect = (start_pos.x + x * size.x, start_pos.y + y * size.y, 
                        size.x, size.y)
                pg.draw.rect(self.render_surface, color, rect)
    

    def render_automata2(self, grid_pattern : np.ndarray, rect : pg.Rect, render_mode = True) -> None:
        start_pos = pg.Vector2(rect.x, rect.y)
        size = pg.Vector2(int(rect.w / grid_pattern.shape[0]), int(rect.h / grid_pattern.shape[1]))

        for x in range(grid_pattern.shape[0]):
            for y in range(grid_pattern.shape[1]):
                color = self.state_to_color(grid_pattern[x][y], render_mode)
                rect = (start_pos.x + x * size.x, start_pos.y + y * size.y, size.x, size.y)
                pg.draw.rect(self.render_surface, color, rect)
    

    def render_psudo3d_view(self, pattern, rect : pg.Rect):
        grid_pattern = np.transpose(pattern, (2, 0, 1))
        # converts the first 4 layers into r g b a
        to_col = lambda x: np.clip(x, 0, 1) * 255
        r_colors = to_col(grid_pattern[0])
        g_colors = to_col(grid_pattern[1])
        b_colors = to_col(grid_pattern[2])
        a_colors = to_col(grid_pattern[3])

        start_pos = pg.Vector2(rect.x, rect.y)
        size = pg.Vector2(int(rect.w / len(grid_pattern[0])), int(rect.h / len(grid_pattern[0][0])))

        for x in range(len(grid_pattern[0])):
            for y in range(len(grid_pattern[0][x])):
                color = (r_colors[x][y], g_colors[x][y], b_colors[x][y], a_colors[x][y])
                rect = (start_pos.x + x * size.x, start_pos.y + y * size.y, 
                        size.x, size.y)
                pg.draw.rect(self.render_surface, color, rect)


    # statistics and calculations    
    def get_avg_of_10(self) -> float: return self.avg_of_10_score
    def get_avg_of_50(self) -> float: return self.avg_of_50_score
    def get_best_score(self) -> float: return self.best_score

    
