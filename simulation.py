import pygame as pg

from settings import *

from evolution_manager import EvolutionManager
from utils.text_drawer import TextDrawer


class Simulation:
    def __init__(self) -> None:
        # user input variables
        self.running = True
        self.paused = False
        self.rendering = True

        # pygame parameters
        self.window = pg.display.set_mode(WINDOW_DIMS, pg.NOFRAME)
        self.clock = pg.time.Clock()
        
        self.evo_manager = EvolutionManager(self.window, TARGET_IMAGE_PATH)

        # graphics initialization
        self.text_renderer = TextDrawer(self.window)
        self.init_bounds()
    

    # calculating the bounderies for the "desired" cellular pattern and the current cellular pattern
    def init_bounds(self):
        dims = self.evo_manager.grid_shape
        max_size = int(500 / max(dims))
        size = pg.Vector2(max_size * dims[0], max_size * dims[1])
        buffer = 50

        half_size = size/2

        # target image (remains the same the whole simulation)
        start_a = pg.Vector2(WINDOW_DIMS[0] - size.x - buffer, buffer)
        self.target_grid_bounds = pg.Rect(start_a.x, start_a.y, size.x, size.y)

        # the current cell states of the best simulation in the generation
        start_b = pg.Vector2(start_a.x, start_a.y + size.y + buffer)
        self.current_grid_bounds = pg.Rect(start_b.x, start_b.y, size.x, size.y)

        # the final state of the best preforming simulation last gen
        y = start_a.y + half_size.y + buffer/2
        self.best_grid_bounds = pg.Rect(buffer, y, half_size.x, half_size.y)

        # subtracting the best_grid by the target_grid to display the difference needed
        self.difference_bounds = pg.Rect(buffer*1.5 + half_size.x, y, half_size.x, half_size.y)
        
        # for the current_grid layer1: Red, layer2: Green, layer3: Blue, layer4: Alpha. then stack them all together
        self.psudo_3d_bounds = pg.Rect(buffer, y + buffer/2 + half_size.y, half_size.x, half_size.y)

    
    def run(self) -> None:
        while self.running:
            self.clock.tick(MAX_FRAME_RATE)
            self.event_manager()
            self.update()

            if self.rendering:
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

        elif event.key == pg.K_s:
            self.evo_manager.save_best(BEST_FILE_LOCATION)
        
        elif event.key == pg.K_l:
            self.evo_manager.load_best(BEST_FILE_LOCATION)


    def update(self) -> None:
        if not self.paused:
            self.evo_manager.tick()
    
    
    def render(self) -> None:
        self.window.fill(WINDOW_COLOR)
        self.display_on_screen_info()

        self.evo_manager.render_desired(self.target_grid_bounds)
        self.evo_manager.render_current_best_gen(self.current_grid_bounds)
        self.evo_manager.render_previous_best_attempt(self.best_grid_bounds)
        self.evo_manager.render_difference_grid(self.difference_bounds)

        self.evo_manager.render_psudo3d_view(self.evo_manager.automatas[0].grid, self.psudo_3d_bounds)

        pg.display.flip()
    

    def display_on_screen_info(self) -> None:
        # general statistics and info
        e = self.evo_manager
        dims = self.evo_manager.grid_shape
        all_cells = dims[0]*dims[1]*dims[2]
        cells = dims[1]*dims[2]
        worst_possible_score = cells

        avg_of_10 = round(e.get_avg_of_10(), 3)
        avg_of_50 = round(e.get_avg_of_50(), 3)
        #score_range = round(e.calc_score_range())

        best_score = round(e.get_best_score(), 3)
        percentage_best = round(best_score / worst_possible_score * 100)

        self.text_renderer.draw_text(WINDOW_TITLE, (50, 50))
        self.text_renderer.draw_text(f"frame rate: {round(self.clock.get_fps(), 2)} fps",           (50, 70))
        self.text_renderer.draw_text(f"CA dims: {dims[0]}x{dims[1]}x{dims[2]} ({all_cells} total)", (50, 90))
        self.text_renderer.draw_text(f"Simulations: {PARRELEL_SIMULATIONS}",                        (50, 110))
        self.text_renderer.draw_text(f"generation {e.gen} (tick {e.ticks} / {TICKS_PER_GENERATION})",      (50, 130))
        self.text_renderer.draw_text(f"average of 10%: {avg_of_10}",                                 (50, 150))
        self.text_renderer.draw_text(f"average of 50%: {avg_of_50}",                                 (50, 170))
        #self.text_renderer.draw_text(f"score range: {score_range}",                                 (50, 190))
        self.text_renderer.draw_text(f"best score: {best_score} ({percentage_best}%)",              (50, 190))