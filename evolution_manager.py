# a class made to find the correct list of weights which will generate an image after N time steps

import numpy as np
import pygame as pg
import json

from PIL import Image

from automata import CellularAutomata
from settings import *

import numpy as np

from operator import itemgetter
from scipy.signal import convolve2d

"""def mutate(parent_weights: np.ndarray, weights: np.ndarray):
    # Create a mask of random True/False values based on mutation rate
    mask = np.random.uniform(0, 1, size=weights.shape) < MUTATION_RATE
    
    # Generate mutation values within the specified range
    mutations = np.random.uniform(-MUTATION_RANGE, MUTATION_RANGE, size=weights.shape)
    
    # Calculate mutated values using the parent weights and mutation mask
    mutated_values = parent_weights + (mask * mutations)
    
    # Apply clamping between 0 and 1 to the mutated values
    mutated_values = np.clip(mutated_values, -1, 1)
    
    # Update weights with the mutated values within the mask
    weights[mask] = mutated_values[mask]
"""


def mutate_array(array, mutation_rate_percent, mutation_range_perent):
    # Convert change and tweak percentages to decimals
    mutation_rate_percent /= 100
    mutation_range_perent /= 100

    # Create a mask to select random elements to modify
    mask = np.random.rand(*array.shape) < mutation_rate_percent

    # Calculate the maximum tweak for each selected element
    max_tweak = array * mutation_range_perent

    # Generate random values for tweaking
    tweaks = np.random.uniform(-max_tweak, max_tweak, size=array.shape)

    # Apply tweaks to the selected elements
    array[mask] += tweaks[mask]

    return array


# Calculate the number of elements to extract based on the percentage
# then Slice the array to get the first N% elements and copy them into a new array
def get_first_n_percent(arr, percent):
    num_elements = int(len(arr) * (percent / 100))
    new_array = arr[:num_elements].copy()
    return new_array


class EvolutionManager:
    def __init__(self, render_surface : pg.Surface, train_image_file_location : str) -> None:
        self.render_surface = render_surface

        r = 6
        l = np.zeros((r, r)) + 1
        self.desired_image = convolve2d(self.load_image(train_image_file_location), l, mode='same') / (r*r)
        shape = self.desired_image.shape
        self.grid_shape = np.array([shape[0], shape[1], AUTOMATA_DEPTH])

        # statistics
        self.gen = 0
        self.ticks = 0
        self.best_score = 0
        self.avg_of_10_score = 0
        self.avg_of_50_score = 0

        self.automatas = self.init_automatas()
        self.calc_grid_diff()
    

    def init_automatas(self) -> list[CellularAutomata]:
        automatas = [CellularAutomata(self.grid_shape) for _ in range(PARRELEL_SIMULATIONS)]

        self.init_grid_states = np.zeros(self.grid_shape)
        idx_a = int(self.grid_shape[0]/2)
        idx_b = int(self.grid_shape[1]/2)

        self.init_grid_states[idx_a][idx_b][0] = 1
        
        for automata in automatas:
            automata.grid = self.init_grid_states.copy()
        
        self.best_grid_saved = automatas[0].grid.copy()

        return automatas


    def calc_grid_diff(self) -> None:
        self.grid_diff = self.desired_image - np.transpose(self.best_grid_saved, (2, 0, 1))[ANALYZE_INDEX]
    
    

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
                color = self.state_to_color(transposed[ANALYZE_INDEX][x][y], render_mode)
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


    def tick(self):
        self.ticks += 1

        func = lambda automata: automata.update_all_states()
        list(map(func, self.automatas))
       
        if (self.ticks >= TICKS_PER_GEN):
            self.start_new_generation()
            self.reset_automatas()


    # Sort automata simulations from best to worst, discarding the bottom half. 
    # The top 10% proceed unchanged. Adjust all scores (including top 10%) to sum up to 1. 
    # Generate a random float between 0 and 1 to select an automata from this pool. 
    # Apply random mutations to the chosen automata to generate a new one. 
    # Repeat this process to form the new population
    def start_new_generation(self):
        sorted_population_indexes, all_scores = self.find_sorted_indexes()

        top_10_percent = get_first_n_percent(sorted_population_indexes, UNMUTATED_PERCENT)
        top_50_percent = get_first_n_percent(sorted_population_indexes, UNDISCARDED_PERCENT)

        top_50_scores = np.array([all_scores[idx] for idx in top_50_percent])

        inverse_scores = 1 / top_50_scores # we want smaller values to be favoured
        inverse_prob_dist = inverse_scores / np.sum(inverse_scores)


        # creating the new population container
        w1_shape = self.automatas[0].l1_weights.shape
        w2_shape = self.automatas[0].l2_weights.shape
        b1_shape = self.automatas[0].l1_biases.shape
        b2_shape = self.automatas[0].l2_biases.shape

        self.l1_new_weights = np.zeros((PARRELEL_SIMULATIONS, w1_shape[0], w1_shape[1]))
        self.l1_new_biases  = np.zeros((PARRELEL_SIMULATIONS, b1_shape[0]))

        self.l2_new_weights = np.zeros((PARRELEL_SIMULATIONS, w2_shape[0], w2_shape[1]))
        self.l2_new_biases  = np.zeros((PARRELEL_SIMULATIONS, b2_shape[0]))

        # re-building the population
        for i in range(PARRELEL_SIMULATIONS):
            # the top 10% move on to the next gen unchanged
            if i < len(top_10_percent):
                automata = self.automatas[top_10_percent[i]]
                self.l1_new_weights[i] = automata.l1_weights.copy()
                self.l2_new_weights[i] = automata.l2_weights.copy()
                self.l1_new_biases[i]  = automata.l1_biases.copy()
                self.l2_new_biases[i]  = automata.l2_biases.copy()
            
            else:
                idx = np.random.choice(top_50_percent, p=inverse_prob_dist)
                parent = self.automatas[idx]
                self.l1_new_weights[i] = mutate_array(parent.l1_weights.copy(), W_MUTATION_RATE, W_MUTATION_RANGE)
                self.l2_new_weights[i] = mutate_array(parent.l2_weights.copy(), W_MUTATION_RATE, W_MUTATION_RANGE)
                self.l1_new_biases[i]  = mutate_array(parent.l1_biases.copy(),  B_MUTATION_RATE, B_MUTATION_RANGE) 
                self.l2_new_biases[i]  = mutate_array(parent.l2_biases.copy(),  B_MUTATION_RATE, B_MUTATION_RANGE) 

        for i in range(PARRELEL_SIMULATIONS):
            self.automatas[i].l1_weights = self.l1_new_weights[i]
            self.automatas[i].l2_weights = self.l2_new_weights[i]
            self.automatas[i].l1_biases = self.l1_new_biases[i]
            self.automatas[i].l2_biases = self.l2_new_biases[i]

        # calculating some statistics
        top_10_scores = np.array([all_scores[idx] for idx in top_10_percent])
        self.avg_of_10_score = sum(top_10_scores) / len(top_10_scores)
        self.avg_of_50_score = sum(top_50_scores) / len(top_50_scores)
        self.best_score = min(top_10_scores)

        self.best_grid_saved = self.automatas[top_10_percent[0]].grid.copy()
    

    def reset_automatas(self):
        self.gen += 1
        self.ticks = 0

        for i in range(PARRELEL_SIMULATIONS):
            self.automatas[i].grid = self.init_grid_states.copy()
        
        self.calc_grid_diff()
    

    def find_sorted_indexes(self) -> tuple[np.ndarray, np.ndarray]:
        calc_score = lambda idx: np.sum((self.desired_image - np.transpose(self.automatas[idx].grid, (2, 0, 1))[ANALYZE_INDEX])**2)
        scores = np.array(list(map(calc_score, range(PARRELEL_SIMULATIONS))))

        # Create an array of tuples containing index and final score for each automata
        indexed_scores = np.array([(index, score) for index, score in enumerate(scores)])

        # Sort the indexes based on final scores
        sorted_indexes = np.array([index for index, _ in sorted(indexed_scores, key=itemgetter(1))], np.int64)

        return sorted_indexes, scores
        


    def load_image(self, filepath) -> np.ndarray:
        img = Image.open(filepath)
        img = img.rotate(90, expand=True)
        # Normalize pixel values between 0 and 1
        img_array = np.array(img, dtype=np.float32)
        img_normalized = (img_array / 255)
        
        return img_normalized


    """ saving and loading """
    # finding and saving the top 10 networks to a json file
    def save_best(self, file_location  : str) -> None:
        top_10_percent = get_first_n_percent(self.find_sorted_indexes()[0], 10)
        top_10_networks = []

        for idx in top_10_percent:
            automata : CellularAutomata = self.automatas[idx]
      
            network_params = {
                "l1_weights" : automata.l1_weights.tolist(),
                "l2_weights" : automata.l2_weights.tolist(),
                "l1_biases"  : automata.l1_biases.tolist(),
                "l2_biases"  : automata.l2_biases.tolist()
            }

            top_10_networks.append(network_params)
        
        with open(file_location, 'w') as file:
            json.dump(top_10_networks, file, indent=4)

    
    # reading the top 10 networks from a json file and generating the next
    def load_best(self, file_location: str) -> None:
        with open(file_location, 'r') as file:
            top_10_networks = json.load(file)
        
        network_data = []

        # Process loaded data to generate the next process
        for network_params in top_10_networks:
            network_data.append((
                np.array(network_params.get('l1_weights')),
                np.array(network_params.get('l2_weights')),
                np.array(network_params.get('l1_biases')),
                np.array(network_params.get('l2_biases'))))
        
        for i in range(PARRELEL_SIMULATIONS):
            a = self.automatas[i]
            a.l1_weights, a.l2_weights, a.l1_biases, a.l2_biases = network_data[i%len(network_data)]
