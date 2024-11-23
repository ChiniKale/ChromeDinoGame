import pygame

class Dinosaur:
    def __init__(self, surface_height):
        self.x = 60
        self.y = 0
        self.width = 40
        self.height = 60  # Standing height
        self.duck_height = 30  # Ducking height
        self.y_velocity = 0
        self.is_jumping = False
        self.is_ducking = False
        self.is_collided = False
        self.surface_height = surface_height

        # Load images
        self.running_frames = [
            pygame.transform.scale(pygame.image.load(r"dinorun0000.png"), (self.width, self.height)),
            pygame.transform.scale(pygame.image.load(r"dinorun0001.png"), (self.width, self.height)),
        ]
        self.jumping_frames = [
            pygame.transform.scale(pygame.image.load(r"dinoJump0000.png"), (self.width, self.height)),
        ]
        self.collision_frames = [
            pygame.transform.scale(pygame.image.load(r"dinoDead0000.png"), (self.width, self.height)),
        ]
        self.ducking_frames = [
            pygame.transform.scale(pygame.image.load(r"dinoduck0000.png"), (self.width * 1.4, self.duck_height)),
            pygame.transform.scale(pygame.image.load(r"dinoduck0001.png"), (self.width * 1.4, self.duck_height)),
        ]

        self.current_frame = 0
        self.animation_time = 0.1  # Time per frame in seconds
        self.time_accumulator = 0  # Tracks elapsed time to switch frames

    def duck(self, is_ducking):
        """Set ducking state."""
        self.is_ducking = is_ducking

    def jump(self):
        """Initiates jump if on the ground."""
        if self.y == 0:
            self.y_velocity = 300
            self.is_jumping = True

    def update(self, delta_time):
        """Updates position and animation."""
        # Apply gravity
        self.y_velocity -= 750 * delta_time
        self.y += self.y_velocity * delta_time

        if self.y < 0:  # Reset on ground contact
            self.y = 0
            self.y_velocity = 0
            self.is_jumping = False

        # Animation frame update
        self.time_accumulator += delta_time
        if self.time_accumulator > self.animation_time:
            self.current_frame = (self.current_frame + 1) % (
                len(self.ducking_frames) if self.is_ducking else len(self.running_frames)
            )
            self.time_accumulator = 0

    def get_collision_rect(self):
        """Returns the collision rectangle, adjusted for ducking."""
        if self.is_ducking:
            return pygame.Rect(self.x, self.surface_height - self.y - self.duck_height, self.width * 1.4, self.duck_height)
        else:
            return pygame.Rect(self.x, self.surface_height - self.y - self.height, self.width, self.height)

    def draw(self, display):
        """Draws the appropriate frame."""
        if self.is_collided:
            current_image = self.collision_frames[0]
        elif self.is_jumping or self.y > 0:
            current_image = self.jumping_frames[0]
        elif self.is_ducking:
            current_image = self.ducking_frames[self.current_frame]
        else:
            current_image = self.running_frames[self.current_frame]

        display.blit(current_image, (self.x, self.surface_height - self.y - self.height))

    def get_rect(self):
        """Returns the rectangle for collision detection."""
        return pygame.Rect(self.x, self.surface_height - self.y - self.height, self.width, self.height)
# 
