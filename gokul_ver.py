import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dinosaur import Dinosaur
from obstacle import Obstacle
from batsymbol import Batsymb
from pygame import mixer

# Initialize Pygame and Mixer
pygame.init()
mixer.init()
clock = pygame.time.Clock()

# Constants
WIDTH, HEIGHT = 640, 480
GROUND_HEIGHT = HEIGHT - 100
FPS = 60
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
MINGAP = 200
MAXGAP = 600
MAXSIZE = 40
MINSIZE = 20

# Load Assets
ground_image = pygame.image.load("ground.png")
ground_width = ground_image.get_width()
mixer.music.load("bgm.mp3")
achievement_sound = mixer.Sound("100.mp3")

# DinoModel Neural Network
class DinoModel(nn.Module):
    def __init__(self):
        super(DinoModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),  # Using LayerNorm for better compatibility
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Actions: [jump, duck, do nothing]
        )

    def forward(self, x):
        return self.fc(x)

# Draw Text Function
def draw_text(text, font, color, x, y, surface):
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect()
    text_rect.topleft = (x, y)
    surface.blit(text_obj, text_rect)

def get_game_state(dinosaur, obstacles, velocity):
    next_obstacle = next((obs for obs in obstacles if obs.x > dinosaur.x), None)
    if next_obstacle:
        is_high = 1 if next_obstacle.y < next_obstacle.GroundHeight - next_obstacle.size else 0
        state = [
            next_obstacle.x - dinosaur.x,
            next_obstacle.y - dinosaur.y,
            velocity,
            is_high,
        ]
    else:
        state = [WIDTH, 0, velocity, 0]
    return np.array(state, dtype=np.float32)

# Game Logic
import time  # Import time for time tracking

# Inside the run_game function
def run_game(agent, memory, optimizer, criterion, epsilon, train=True):
    game_display = pygame.display.set_mode((WIDTH, HEIGHT))
    dinosaur = Dinosaur(GROUND_HEIGHT)
    bat = Batsymb(0, 115)
    obstacles = []
    lastObstacle = WIDTH
    ground_scroll = 0
    game_timer, score = 0, 0
    velocity = 300

    last_action_time = time.time()  # Initialize the last action time

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
        draw_text(f"Score: {score}", pygame.font.SysFont("Helvetica", 30), (0, 255, 0), 50, 50, game_display)

        # Spawn Obstacles
        if len(obstacles) == 0 or obstacles[-1].x < WIDTH - MINGAP:
            is_high = random.random() > 0.7
            obstacle_size = random.randint(MINSIZE, MAXSIZE) if not is_high else 30
            obstacles.append(Obstacle(lastObstacle, obstacle_size, GROUND_HEIGHT, is_high))
            lastObstacle += MINGAP + (MAXGAP - MINGAP) * random.random() + 0.01 * game_timer * 1000

        # Update Bat
        bat.update(delta_time)
        bat.draw(game_display)

        # Current State
        state = get_game_state(dinosaur, obstacles, velocity)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Decide Action
        current_time = time.time()
        if train and random.random() < epsilon:
            action = random.choice([0, 1, 2])  # Exploration
        else:
            with torch.no_grad():
                agent.eval()  # Set model to evaluation mode
                action_prob = agent(state_tensor)
                agent.train()  # Restore training mode
                action = torch.argmax(action_prob).item()  # Exploitation

        # Perform Action if 100ms has passed since last action
        if current_time - last_action_time >= 0.15:  # 100 ms = 0.1 seconds
            if action == 0:
                dinosaur.bigjump()
            elif action == 1:
                dinosaur.duck(True)
            else:
                dinosaur.duck(False)

            last_action_time = current_time  # Update the last action time

        # Update and Draw Obstacles
        for obs in obstacles:
            obs.update(delta_time, velocity)
            obs.draw(game_display)
            dino_rect = pygame.Rect(dinosaur.x, GROUND_HEIGHT - dinosaur.y - dinosaur.height, dinosaur.width, dinosaur.height)
            obs_rect = pygame.Rect(obs.x, obs.y, obs.size, obs.size)

            if dino_rect.colliderect(obs_rect):
                mixer.music.stop()
                running = False

        lastObstacle -= velocity * delta_time

        # Update Dinosaur
        dinosaur.update(delta_time)
        dinosaur.draw(game_display)

        # Update Memory for RL
        next_state = get_game_state(dinosaur, obstacles, velocity)
        reward = -1 if not running else 0.1
        memory.append((state, action, reward, next_state, not running))

        # Train Agent
        if train and len(memory) > 32:
            batch = random.sample(memory, 32)
            states, actions, rewards, next_states, dones = zip(*batch)

            states_tensor = torch.tensor(states, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.long)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32)

            q_values = agent(states_tensor)
            next_q_values = agent(next_states_tensor)
            target_q_values = q_values.clone()

            for i in range(len(batch)):
                target_q_values[i, actions_tensor[i]] = rewards_tensor[i] + (0.99 * torch.max(next_q_values[i]) * (1 - dones[i]))

            loss = criterion(q_values, target_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pygame.display.update()

    return score


if __name__ == "__main__":
    agent = DinoModel()
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    memory = []
    epsilon = 0.01  # Exploration rate

    scores = []
    for episode in range(1000):
        score = run_game(agent, memory, optimizer, criterion, epsilon, train=True)
        scores.append(score)
        print(f"Episode {episode + 1}: Score = {score}")

    torch.save(agent.state_dict(), "dino_model.pth")
    print("Training Complete. Model saved.")
