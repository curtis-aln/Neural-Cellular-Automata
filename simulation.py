import pygame as pg

from settings.settings import *

from population import EvolutionManager, get_desired_and_init_states
from utils.text_drawer import TextDrawer
from utils.neat_trainer import NeatFileManager
from scipy.ndimage import zoom
from automata import CellularAutomata


def array_to_colored_surface(array, target_width, target_height):
    # Normalize the pixel values to the range [0, 255]
    normalized_image = (array * 255).astype(np.uint8)

    # Create a color gradient (e.g., magenta to blue)
    def color_gradient(value):
        magenta = np.array([255, 0, 255])
        blue = np.array([0, 0, 255])
        return (magenta * (1 - value) + blue * value).astype(np.uint8)

    # Apply the color gradient to the normalized image
    colored_image = np.array([color_gradient(value / 255) for value in normalized_image.flatten()]).reshape(
        (normalized_image.shape[0], normalized_image.shape[1], 3))

    # Compute the zoom factors
    zoom_factor_y = target_height / colored_image.shape[1]
    zoom_factor_x = target_width / colored_image.shape[0]

    # Resize the colored image using the computed zoom factors
    resized_colored_image = zoom(colored_image, (zoom_factor_y, zoom_factor_x, 1), order=1)

    # Create a surface from the colored image
    surface = pg.surfarray.make_surface(resized_colored_image)
    return surface


# The run best automata class will take the best preforming automata from the save file and bring it into an
# interactive pygame simulation where the user can play with and test the results from training.
class RunBestAutomata(NeatFileManager):
    def __init__(self, file_config_path = "", population_save_file = "", best_genome_save_file = "") -> None:
        super().__init__(file_config_path, population_save_file, best_genome_save_file)

        # loading data from files
        self.best_genome = self.get_best_genome_from_file()
        self.neural_network = self.genome_to_network(self.best_genome)
        
        self.desired_image, self.image_shape, self.init_grid_states = get_desired_and_init_states()

        self.automata = CellularAutomata(self.init_grid_states)
    

    def render(self, surface) -> None:
        bounds = (0, 0, 1280, 720)

        automata_surface = array_to_colored_surface(self.automata.get_image(), bounds[2], bounds[3])
        surface.blit(automata_surface, bounds)
    

    def update(self) -> None:
        self.automata.update_all_states(self.neural_network)



class Simulation:
    def __init__(self) -> None:
        # user input variables
        self.running = True
        self.paused = False
        self.rendering = True

        # pygame parameters
        self.window = pg.display.set_mode(WINDOW_DIMS, pg.NOFRAME)
        self.clock = pg.time.Clock()
        
        # graphics initialization
        self.text_renderer = TextDrawer(self.window)

        self.run_best = RunBestAutomata(CONFIG_PATH, SAVE_FILE_POPULATION, SAVE_FILE_BEST_GENOME)
    
    def run(self) -> None:
        while self.running:
            self.clock.tick(MAX_FRAME_RATE)
            self.event_manager()
            self.update()

            self.render()


    def event_manager(self) -> None:
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                self.key_press_event(event)
    

    def key_press_event(self, event):
        if event.key == pg.K_ESCAPE:
            self.running = False
        
        elif event.key == pg.K_SPACE:
            self.paused = not self.paused
        
        elif event.key == pg.K_o:
            self.rendering = not self.rendering


    def update(self) -> None:
        if not self.paused:
            self.run_best.update()
    

    def render(self) -> None:
        self.window.fill(WINDOW_COLOR)

        self.run_best.render(self.window)

        pg.display.flip()
