import numpy as np

TARGET_IMAGE_PATH = "images/astro_compressed.jpg"
BEST_FILE_LOCATION = "data/data.json"


""" pygame parameters """
WINDOW_COLOR = (0, 0, 0)
WINDOW_TITLE = "Neural Cellular Automata"
WINDOW_DIMS = (1550, 850)
MAX_FRAME_RATE = 600

""" graphics parameters """


""" evolution paramaters """
PARRELEL_SIMULATIONS = 400 # automatas running in parralel
TICKS_PER_GENERATION = 80  # frames every generation


""" cellular automata parameters """
AUTOMATA_DEPTH = 8     # the amount of third dimensional layers per sim
ANALYZE_LAYER  = 0      # layer used to calculate score


""" evolution settings """
B_MUTATION_RATE  = 15#%
B_MUTATION_RANGE = 15#%
W_MUTATION_RATE  = 15#%
W_MUTATION_RANGE = 15#%

UNMUTATED_PERCENT   = 10 # percent of the population that go mutated
UNDISCARDED_PERCENT = 50 # percent of the population that isn't discarded

L1_W_RANGE = 0.04 # range of weight values allowed to initilise layer 1
L2_W_RANGE = 0.04 # range of weight values allowed to initilise layer 2
L1_B_RANGE = 0.0 # range of bias values allowed to initilise layer 1
L2_B_RANGE = 0.0 # range of bias values allowed to initilise layer 2


SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]])
SOBEL_Y = SOBEL_X.T

SOBEL_Z = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
])

# making them 3d matricies
SOBEL_X = np.transpose(SOBEL_X[None,:, :], (1, 2, 0))
SOBEL_Y = np.transpose(SOBEL_Y[None,:, :], (1, 2, 0))


SAVE_FREQ = 5
PRINT_FREQ = 5
CHANGE_TARGETS_FREQ = 50
SAVE_FILE_POPULATION = ''#'data/neat_population.pkl'
SAVE_FILE_BEST_GENOME = 'data/best_genome.pkl'
CONFIG_PATH = "settings/config-feedforward"