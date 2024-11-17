import pygame
import random

colour = 0, 0, 255

class Obstacle:
    def __init__(self, x, size, GroundHeight, is_high=True):
        self.x = x
        self.size = size
        self.GroundHeight = GroundHeight
        if is_high:
            self.y = GroundHeight - size - 30  # Height of the sky obstacles
        else:
            self.y = GroundHeight - size  # Ground obstacles

    def draw(self, gameDisplay):
        pygame.draw.rect(gameDisplay, colour, [self.x, self.y, self.size, self.size])
    
    def update(self, deltaTime, velocity):
        self.x -= velocity * deltaTime

    def checkOver(self):
        return self.x < 0
