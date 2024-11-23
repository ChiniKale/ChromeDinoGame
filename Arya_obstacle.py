import pygame

class Obstacle:
    def __init__(self, x, size, ground_height, is_high=False):
        self.x = x
        self.size = size
        self.is_high = is_high
        self.ground_height = ground_height
        self.frames = [
            pygame.transform.scale(pygame.image.load("birb.png" if is_high else "cactus.png"), (size, size))
        ]
        if is_high:
            self.frames.append(pygame.transform.scale(pygame.image.load("birb2.png"), (size, size)))

        self.y = ground_height - size if not is_high else ground_height - size - 30
        self.current_frame = 0
        self.animation_time = 0.5
        self.time_accumulator = 0

    def update(self, delta_time, velocity):
        """Updates obstacle position and animation."""
        self.x -= velocity * delta_time
        self.time_accumulator += delta_time
        if self.time_accumulator > self.animation_time and len(self.frames) > 1:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.time_accumulator = 0

    def draw(self, display):
        """Draws the obstacle."""
        current_image = self.frames[self.current_frame]
        display.blit(current_image, (self.x, self.y))

    def get_rect(self):
        """Returns collision rectangle."""
        return pygame.Rect(self.x, self.y, self.size, self.size)

    def is_off_screen(self):
        """Check if off-screen."""
        return self.x < -self.size
