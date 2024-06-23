import numpy as np

TARGET_IMAGE_PATH = "images/astro_compressed.jpg"
BEST_FILE_LOCATION = "data/data.json"


""" pygame parameters """
WINDOW_COLOR = (0, 0, 0)
WINDOW_TITLE = "Neural Cellular Automata"
WINDOW_DIMS = (1350, 800)
MAX_FRAME_RATE = 600



""" cellular automata parameters """
AUTOMATA_DEPTH = 6     # the amount of third dimensional layers per sim
ANALYZE_LAYER  = 0      # layer used to calculate score


""" evolution settings """
POPULATION_SIZE = 20
GENERATIONS = 100
ITERATIONS = 30


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