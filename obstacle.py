import pygame

colour = 0,0,255
class Obstacle:
    def __init__(self, x, size, GroundHeight,  is_high=False):
        self.x = x
        self.size = size
        self.GroundHeight = GroundHeight
        self.y = GroundHeight - size if not is_high else GroundHeight - size - 60

    def draw(self, gameDisplay):
        pygame.draw.rect(gameDisplay, colour, [self.x, self.GroundHeight-self.size, self.size, self.size])
    
    def update(self, deltaTime, velocity):
        self.x -= velocity*deltaTime

    def checkOver(self):
        if self.x < 0:
            return True
        else:
            return False
        

        