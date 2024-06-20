import pygame as pg

pg.font.init()

class TextDrawer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pg.font.Font(None, 28)  # You can choose any font and size here

    def draw_text(self, text, position, color=(255, 255, 255)):
        rendered_text = self.font.render(text, True, color)
        self.screen.blit(rendered_text, position)