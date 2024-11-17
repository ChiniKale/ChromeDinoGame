import pygame
import random

class Batsymb:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.active = False
        self.image = pygame.image.load("Bat symbol.png")
        self.image = pygame.transform.scale(self.image, (260, 260))  # Scale cloud size
        self.timer = random.randint(5, 10)

    def update(self, deltaTime, velocity=100):
        if self.active:
            self.x -= velocity * deltaTime  # Move the symbol
            if self.x < -self.image.get_width():  # If it moves off-screen
                self.active = False  # Deactivate
                self.timer = random.randint(5, 10)  # Reset the timer

        else:  # Decrement timer when inactive
            self.timer -= deltaTime
            if self.timer <= 0:  # Reactivate after the timer runs out
                self.x = 800 + random.randint(0, 300)  # Reset position
                self.y = 115  # Randomize vertical position
                self.active = True

    def draw(self, gameDisplay):
        if self.active:    
            gameDisplay.blit(self.image, (self.x, self.y))
