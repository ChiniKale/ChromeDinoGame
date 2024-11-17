import pygame

dinocolour = 255,255,255
DINOHEIGHT = 45
DINOWIDTH = 20
FPS = 60

class Dinosaur:
    def __init__(self, surfaceHeight):
        self.x = 60
        self.y = 10
        self.height = 60  # New height of the dinosaur
        self.width = 30
        self.yvelocity = 0
        self.is_collided = False

        size = (self.width, self.height)

        self.running_frames = [
            pygame.transform.scale(pygame.image.load(r"dinorun0000.png"), size),
            pygame.transform.scale(pygame.image.load(r"dinorun0001.png"), size),
        ]
        self.jumping_frames = [
            pygame.transform.scale(pygame.image.load(r"dinoJump0000.png"), size),
        ]
        self.collision_frames = [
            pygame.transform.scale(pygame.image.load(r"dinoDead0000.png"), size),
        ]    
        self.current_frame = 0
        self.animation_time = 0.1  # Time per frame in seconds
        self.time_accumulator = 0  # Tracks elapsed time to switch frames
        self.height = DINOHEIGHT
        self.width = DINOWIDTH
        self.surfaceHeight = surfaceHeight
        self.is_jumping = False  # Indicates if the dinosaur is jumping
    def update_collision_animation(self, deltaTime):
        # Update animation frame if collision occurs
        if self.is_collided:
            self.time_accumulator += deltaTime
            if self.time_accumulator > self.animation_time:
                self.current_frame += 1
                self.time_accumulator = 0
            if self.current_frame >= len(self.collision_frames):
                return True  # Animation is complete
        return False    
    def jump(self): #When adding classes into function, the first parameter must be the parameter
        if(self.y == 10): #Only allow jumping if the dinosaur is on the ground to prevent mid air jumps.
            self.yvelocity = 300
            self.is_jumping = True
    def update(self, deltaTime): #Updates the y position of the dinosaur each second
        self.yvelocity += -1000*deltaTime #Gravity
        self.y += self.yvelocity * deltaTime
        if self.y < 10: #if the dinosaur sinks into the ground, make velocity and y = 0
            self.y = 10
            self.yvelocity = 0
            self.is_jumping = False
        self.time_accumulator += deltaTime
        if self.time_accumulator > self.animation_time:
            self.current_frame = (self.current_frame + 1) % len(self.running_frames)
            self.time_accumulator = 0    

        
    def draw(self, display):
        if self.is_jumping or self.y > 10:
            current_image = self.jumping_frames[self.current_frame % len(self.jumping_frames)]
        elif  self.is_collided:
            current_image = self.collision_frames[self.current_frame % len(self.collision_frames)]    
        else:
            current_image = self.running_frames[self.current_frame % len(self.running_frames)]

        display.blit(current_image, (self.x, self.surfaceHeight - self.y - self.height))    
