import pygame
from dinosaur import Dinosaur  # Import the class Dinosaur from the file ’dinosaur’
from obstacle_new import Obstacle
from batsymbol import Batsymb
import random
import torch
import torch.nn as nn
import numpy as np

pygame.init()  # Start pygame
clock = pygame.time.Clock()
from pygame import mixer

# Starting the mixer
mixer.init()
mixer.music.load("bgm.mp3")  # Load background music
mixer.music.set_volume(0.7)
mixer.music.play(loops=-1)

# Define constants
size = width, height = 640, 480
GROUND_HEIGHT = height - 100
MINGAP = 200
MAXGAP = 600
MAXSIZE = 40
MINSIZE = 20
WIDTH = 640

# Initialize game components
gameDisplay = pygame.display.set_mode(size)
dinosaur = Dinosaur(GROUND_HEIGHT)
Bat = Batsymb(0, 115)
ground_image = pygame.image.load(r"ground.png")
ground_width = ground_image.get_width()
ground_scroll = 0
obstacles = []
lastObstacle = width
game_timer = 0
lastFrame = pygame.time.get_ticks()

# Colors and fonts
text_font = pygame.font.SysFont("Helvetica", 30)
white = (255, 255, 255)

# Neural network for RL agent
class DinoModel(nn.Module):
    def __init__(self):
        super(DinoModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 128),  # Increased size for initial layer
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Output: [jump, duck, do nothing]
        )

    def forward(self, x):
        return self.fc(x)

# Load trained model
q_network = DinoModel()
q_network.load_state_dict(torch.load("dino_model.pth"))
q_network.eval()

def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    gameDisplay.blit(img, (x, y))

def reset_game():
    global game_timer, lastFrame, dinosaur, obstacles, lastObstacle, ground_scroll
    game_timer = 0
    lastFrame = pygame.time.get_ticks()
    dinosaur = Dinosaur(GROUND_HEIGHT)
    obstacles = []
    lastObstacle = width
    ground_scroll = 0
    mixer.music.load("bgm.mp3")
    mixer.music.play(loops=-1)

def get_game_state(dinosaur, obstacles, velocity):
    next_obstacles = [obs for obs in obstacles if obs.x > dinosaur.x]
    if len(next_obstacles) > 0:
        next_obstacle = next_obstacles[0]
        # second_obstacle = next_obstacles[1] if len(next_obstacles) > 1 else None

        # Additional state features
        state = [
            next_obstacle.x - dinosaur.x,  # Distance to next obstacle
            next_obstacle.y - dinosaur.y,  # Vertical distance
            next_obstacle.size,            # Size of next obstacle
            # second_obstacle.x - dinosaur.x if second_obstacle else WIDTH,  # Distance to second obstacle
            velocity,                      # Current game velocity
            1 if dinosaur.is_jumping else 0,  # Is the dinosaur jumping?
            1 if dinosaur.is_ducking else 0,  # Is the dinosaur ducking?
        ]
    else:
        # Default state when no obstacles are nearby
        state = [WIDTH, 0, 0, velocity, 0, 0]

    return np.array(state, dtype=np.float32)

while True:  # Main game loop
    t = pygame.time.get_ticks()
    deltaTime = (t - lastFrame) / 1000.0
    lastFrame = t
    game_timer += deltaTime
    VELOCITY = 300 + 0.01 * game_timer * 1000

    # Get game state and make a decision using the RL agent
    state = get_game_state(dinosaur, obstacles, VELOCITY)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(state_tensor)
    action = q_values.argmax().item()

    # Perform the chosen action
    if action == 1:  # Jump
        dinosaur.bigjump()
    elif action == 2:  # Duck
        dinosaur.duck(True)
    else:  # Do Nothing
        dinosaur.duck(False)

    # Update display and game objects
    gameDisplay.fill((255, 255, 255))
    ground_scroll -= VELOCITY * deltaTime
    if ground_scroll <= -ground_width:
        ground_scroll += ground_width

    gameDisplay.blit(ground_image, (ground_scroll, 300))
    gameDisplay.blit(ground_image, (ground_scroll + ground_width, 300))

    # Draw score
    draw_text(f"Score: {int(game_timer*10)}", text_font, (0, 255, 0), 100, 50)

    dinosaur.update(deltaTime)
    dinosaur.draw(gameDisplay)
    Bat.update(deltaTime)
    Bat.draw(gameDisplay)

    if len(obstacles) == 0 or obstacles[-1].x < width - MINGAP:
        is_high = random.random() > 0.7
        obstacle_size = random.randint(MINSIZE, MAXSIZE) if not is_high else 30
        obstacles.append(Obstacle(lastObstacle, obstacle_size, GROUND_HEIGHT, is_high))
        lastObstacle += MINGAP + (MAXGAP - MINGAP) * random.random() + 0.01 * game_timer * 1000

    # Check for collisions and update obstacles
    for obs in obstacles:
        obs.update(deltaTime, VELOCITY)
        obs.draw(gameDisplay)

        dino_rect = pygame.Rect(dinosaur.x, dinosaur.surfaceHeight - dinosaur.y - dinosaur.height, dinosaur.width, dinosaur.height)
        obs_rect = pygame.Rect(obs.x, obs.y, obs.size, obs.size)

        if dino_rect.colliderect(obs_rect):
            mixer.music.load("gameover.mp3")
            mixer.music.set_volume(0.7)
            mixer.music.play()
            reset_game()

    lastObstacle -= VELOCITY * deltaTime
    pygame.display.update()
    clock.tick(60)
