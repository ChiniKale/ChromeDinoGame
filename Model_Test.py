import pygame
from dinosaur import Dinosaur  # Import the class Dinosaur from the file ’dinosaur’
from obstacle import Obstacle
from batsymbol import Batsymb
import random
import torch
import torch.nn as nn

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
class DinoAgent(nn.Module):
    def __init__(self):
        super(DinoAgent, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # 3 actions: [Do nothing, Jump, Duck]

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load trained model
q_network = DinoAgent()
q_network.load_state_dict(torch.load("dino_q_model.pth"))
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
    if obstacles:
        next_obstacle = obstacles[0]
        distance = next_obstacle.x - dinosaur.x
        size = next_obstacle.size
        is_high = int(next_obstacle.is_high)
    else:
        distance = size = is_high = 0
    return [distance, size, is_high, dinosaur.y, velocity]

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
