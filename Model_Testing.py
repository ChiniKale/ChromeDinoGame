import pygame
import random
import torch
from dinosaur import Dinosaur
from obstacle_new import Obstacle
from batsymbol import Batsymb
from pygame import mixer
import torch.nn as nn

# Initialize Pygame and Mixer
pygame.init()
mixer.init()
clock = pygame.time.Clock()

# Constants
WIDTH, HEIGHT = 640, 480
GROUND_HEIGHT = HEIGHT - 100
FPS = 60
WHITE = (255, 255, 255)
MINGAP = 200
MAXGAP = 600
MAXSIZE = 40
MINSIZE = 20

# Load Assets
ground_image = pygame.image.load("ground.png")
ground_width = ground_image.get_width()
mixer.music.load("bgm.mp3")
mixer.music.play(-1)

# DinoModel Neural Network
# class DinoModel(nn.Module):
#     def __init__(self):
#         super(DinoModel, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(6, 256),  # Increased input size and more neurons
#             nn.ReLU(),
#             nn.Dropout(0.2),  # Prevent overfitting
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 3)  # Output: [jump, duck, do nothing]
#         )
#         # Softmax will be applied outside this model during action selection.

#     def forward(self, x):
#         return self.fc(x)

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
            nn.Linear(32, 3)  # Output: [jump, duck,do nothing]
        )

    def forward(self, x):
        return self.fc(x)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enhanced State Representation
def get_game_state(dinosaur, obstacles, velocity):
    next_obstacles = [obs for obs in obstacles if obs.x > dinosaur.x]
    if len(next_obstacles) > 0:
        next_obstacle = next_obstacles[0]
        state = [
            next_obstacle.x - dinosaur.x,  # Distance to next obstacle
            next_obstacle.y - dinosaur.y,  # Vertical distance
            next_obstacle.size,            # Size of next obstacle
            velocity,                      # Current game velocity
            1 if dinosaur.is_jumping else 0,  # Is the dinosaur jumping?
            1 if dinosaur.is_ducking else 0,  # Is the dinosaur ducking?
        ]
    else:
        state = [WIDTH, 0, 0, velocity, 0, 0]

    return torch.tensor(state, dtype=torch.float32).to(device)

# Game Logic
def run_game(agent):
    game_display = pygame.display.set_mode((WIDTH, HEIGHT))
    dinosaur = Dinosaur(GROUND_HEIGHT)
    bat = Batsymb(0, 115)
    obstacles = []
    last_obstacle = WIDTH+500
    ground_scroll = 0
    game_timer = 0

    running = True
    while running:
        delta_time = clock.tick(FPS) / 1000.0
        game_timer += delta_time
        velocity = 300 + 0.01 * game_timer * 1000  # Dynamic difficulty

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Update Ground
        ground_scroll -= velocity * delta_time
        if ground_scroll <= -ground_width:
            ground_scroll += ground_width

        game_display.fill(WHITE)
        game_display.blit(ground_image, (ground_scroll, 300))
        game_display.blit(ground_image, (ground_scroll + ground_width, 300))

        # Score Display
        score = int(game_timer * 10)
        font = pygame.font.SysFont("Helvetica", 30)
        text_surface = font.render(f"Score: {score}", True, (0, 255, 0))
        game_display.blit(text_surface, (50, 50))

        # Spawn Obstacles
        if len(obstacles) == 0 or obstacles[-1].x < WIDTH - MINGAP:
            is_high = random.random() > 0.7
            obstacle_size = random.randint(MINSIZE, MAXSIZE) if not is_high else 30
            obstacles.append(Obstacle(last_obstacle, obstacle_size, GROUND_HEIGHT, is_high))
            last_obstacle += MINGAP + (MAXGAP - MINGAP) * random.random() + 0.01 * game_timer * 1000

        # Update Bat
        bat.update(delta_time)
        bat.draw(game_display)

        # Current State
        state = get_game_state(dinosaur, obstacles, velocity).unsqueeze(0)

        # Decide Action
        with torch.no_grad():
            action_prob = agent(state)
            action = torch.argmax(action_prob).item()

        # Perform Action
        if action == 0:
            dinosaur.bigjump()
        elif action == 1:
            dinosaur.duck(True)
        else:
            dinosaur.duck(False)

        # Update and Draw Obstacles
        for obs in obstacles:
            obs.update(delta_time, velocity)
            obs.draw(game_display)
            dino_rect = pygame.Rect(dinosaur.x, GROUND_HEIGHT - dinosaur.y - dinosaur.height, dinosaur.width, dinosaur.height)
            obs_rect = pygame.Rect(obs.x, obs.y, obs.size, obs.size)

            if dino_rect.colliderect(obs_rect):
                mixer.music.stop()
                running = False

        last_obstacle -= velocity * delta_time

        # Update Dinosaur
        dinosaur.update(delta_time)
        dinosaur.draw(game_display)

        pygame.display.update()

    print(f"Game Over! Score: {score}")

# Main Program
if __name__ == "__main__":
    agent = DinoModel().to(device)
    agent.load_state_dict(torch.load("dino_genetic_model.pth"))
    run_game(agent)
