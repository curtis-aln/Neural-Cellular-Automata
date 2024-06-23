import numpy as np

from scipy.signal import convolve
from settings.settings import *

from neat.nn import FeedForwardNetwork as Network


class CellularAutomata:
    def __init__(self, inital_grid : np.ndarray) -> None:    
        self.initial_states = inital_grid.copy()
        self.grid = inital_grid.copy()
        self.score = 0


    def update_all_states(self, neural_network : Network) -> None:
        grid_width, grid_height, grid_depth = self.grid.shape

        # we apply a 2d convelution to every cell inside the automata, twice. once for sobel X and one for sobel Y
        grad_x = convolve(self.grid, SOBEL_X, mode='same')
        grad_y = convolve(self.grid, SOBEL_Y, mode='same')
        grad_z = self.grid

        outputs = np.zeros(self.grid.shape)
        for x in range(grid_width):
            for y in range(grid_height):
                inputs = np.concatenate((grad_x[x,y], grad_y[x,y], grad_z[x,y]))
                outputs[x][y] = neural_network.activate(inputs)
        
        random_mask = np.random.rand(*outputs.shape) < 0.5
        random_mask = random_mask.astype(np.float32)
        ds_grid = outputs * random_mask

        self.grid = np.tanh(self.grid + ds_grid)
    

    def run(self, iterations : int, neural_network : Network, desired_image : np.ndarray) -> float:
        self.grid = self.initial_states.copy()

        # running the automata for the given number of iterations
        for i in range(iterations):
            self.update_all_states(neural_network)

        # calculating the score by subtracting the difference squared from the max difference squared  
        image = self.get_image()      
        difference_squared = (desired_image - image) ** 2

        max_diff = 2 * image.shape[0] * image.shape[1]

        self.score = max_diff - np.sum(difference_squared)
        return self.score


    def get_image(self) -> np.ndarray:
        return np.transpose(self.grid, (2, 0, 1))[ANALYZE_LAYER]


        