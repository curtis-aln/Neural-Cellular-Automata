import numpy as np

from scipy.signal import convolve
from settings import *

# activation functions
def relu(x):  return np.maximum(0, x)
def swish(x): return 1 / (1 + np.exp(-x))
def tanh(x):  return np.tanh(x)



class CellularAutomata:
    def __init__(self, cell_grid_shape : np.ndarray) -> None:
        self.cell_grid_shape = cell_grid_shape
    
        self.grid = np.random.uniform(0, 1, self.cell_grid_shape)

        self.net_shape = (AUTOMATA_DEPTH*3, AUTOMATA_DEPTH*2, AUTOMATA_DEPTH)

        # l1_weights and l2_weights are pre-transposed
        self.l1_weights = np.random.uniform(-L1_W_RANGE, L1_W_RANGE, (self.net_shape[1], self.net_shape[0])).T
        self.l2_weights = np.random.uniform(-L2_W_RANGE, L2_W_RANGE, (self.net_shape[2], self.net_shape[1])).T
        self.l1_biases  = np.random.uniform(-L1_B_RANGE, L1_B_RANGE, self.net_shape[1])
        self.l2_biases  = np.random.uniform(-L2_B_RANGE, L2_B_RANGE, self.net_shape[2])

        # tracking for backpropagation
        self.l1_inputs : np.ndarray
        self.l1_outputs : np.ndarray
        self.l2_outputs : np.ndarray


    def update_all_states(self):
        grid_width, grid_height, _ = self.grid.shape

        # we apply a 2d convelution to every cell inside the automata, twice. once for sobel X and one for sobel Y
        grad_x   = convolve(self.grid, SOBEL_X, mode='same')
        grad_y   = convolve(self.grid, SOBEL_Y, mode='same')
        #grad_z   = convolve(self.grid, SOBEL_Z, mode='same')
        
        self.l1_inputs = np.concatenate((grad_x, grad_y, self.grid), axis=2)
        
        # first layer  
        reshaped_grid = self.l1_inputs.reshape(grid_width * grid_height, self.net_shape[0])
        dot_product = reshaped_grid @ self.l1_weights
        self.l1_outputs = relu(dot_product + self.l1_biases)

        # second layer
        dot_product = np.dot(self.l1_outputs, self.l2_weights)
        self.l2_outputs = dot_product + self.l2_biases
       
        # todo
        # reshaping the outputs to fit the new grid
        reshaped = self.l2_outputs.reshape(grid_width, grid_height, self.net_shape[2])
        
        rand_mask = np.random.rand(*reshaped.shape) < 0.5
        rand_mask = rand_mask.astype(np.float32)
        ds_grid = reshaped * rand_mask

        self.grid += ds_grid
        