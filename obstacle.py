import pygame
import random

colour = 0, 0, 255

class Obstacle:
    def __init__(self, x, size, GroundHeight, is_high=True):
        self.x = x
        self.size = size
        self.GroundHeight = GroundHeight
        self.current_frame = 0
        self.animation_time = 0.5  # Time per frame in seconds
        self.time_accumulator = 0 
        if is_high:
            ahaha = random.random()
            if ahaha >= 0.33:
                self.y = GroundHeight - size - 30
                self.frames = [pygame.transform.scale(pygame.image.load(r"birb.png"), (size, size)), pygame.transform.scale(pygame.image.load(r"birb2.png"), (size, size))]  # Height of the sky obstacles
            else:
                self.y = GroundHeight - size
                self.frames = [pygame.transform.scale(pygame.image.load(r"birb.png"), (size, size)), pygame.transform.scale(pygame.image.load(r"birb2.png"), (size, size))]  # Height of the sky obstacles
        else:
            self.y = GroundHeight - size
            self.frames = [pygame.transform.scale(pygame.image.load(r"cactus.png"), (size, size)), pygame.transform.scale(pygame.image.load(r"cactus.png"), (size, size))]  # Ground obstacles
    
    def draw(self, gameDisplay):
        current_image = self.frames[self.current_frame % len(self.frames)]
        gameDisplay.blit(current_image, (self.x, self.y))
    
    def update(self, deltaTime, velocity):
        self.x -= velocity * deltaTime
        self.time_accumulator += deltaTime
        if self.time_accumulator > self.animation_time:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.time_accumulator = 0   

    def checkOver(self):
        return self.x < 0
