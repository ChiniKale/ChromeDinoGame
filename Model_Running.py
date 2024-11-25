import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dinosaur import Dinosaur
from obstacle_new import Obstacle
from batsymbol import Batsymb
from pygame import mixer

# Initialize Pygame and Mixer
pygame.init()
mixer.init()
clock = pygame.time.Clock()

# Constants
WIDTH, HEIGHT = 640, 480
GROUND_HEIGHT = HEIGHT -100
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
# gameover_sound = mixer.Sound("gameover.mp3")

# class DinoModel(nn.Module):
#     def __init__(self):
#         super(DinoModel, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(10, 256),  # Increased input size and more neurons
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

# DinoModel Neural Network
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

# Simpler Model 
# class DinoModel(nn.Module):
#     def __init__(self):
#         super(DinoModel, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(6, 64),  # Updated input size: 4
#             nn.ReLU(),
#             nn.Linear(64, 3)  # Output: [jump, duck]
#         )

#     def forward(self, x):
#         return self.fc(x)


# Draw Text Function
def draw_text(text, font, color, x, y, surface):
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect()
    text_rect.topleft = (x, y)
    surface.blit(text_obj, text_rect)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enhanced State Representation
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



def calculate_reward(dinosaur, obstacles, running):
    if not running:
        return -50  # Large penalty for game over

    reward = 0.1  # Default survival reward

    for obs in obstacles:
        # If the obstacle is passed, provide a bonus and mark it as passed
        if obs.x < dinosaur.x and not getattr(obs, "passed", False):
            obs.passed = True  # Mark the obstacle as passed

            if obs.y < dinosaur.surfaceHeight - obs.size:  # High obstacle
                if dinosaur.is_ducking:
                    reward += 20  # Bonus for ducking under a high obstacle
                else:
                    reward -= 10  # Penalty for missing a duck opportunity
            else:  # Ground obstacle
                if dinosaur.is_jumping:
                    reward += 10  # Bonus for jumping over a ground obstacle
                else:
                    reward -= 5  # Penalty for not jumping

    # Penalize unnecessary actions when no obstacles are near
    if len(obstacles) > 0:
        next_obstacle = obstacles[0]
        distance_to_next = next_obstacle.x - dinosaur.x
        if distance_to_next > 100:  # No immediate obstacle
            if dinosaur.is_jumping or dinosaur.is_ducking:
                reward -= 2  # Slight penalty for unnecessary actions
        elif 0 < distance_to_next <= 100:  # Obstacle is close
            if not (dinosaur.is_jumping or dinosaur.is_ducking):
                reward -= 2  # Penalty for no action near an obstacle

    return reward

# Game Logic
def run_game(agent, memory, optimizer, criterion, train=True):
    game_display = pygame.display.set_mode((WIDTH, HEIGHT))
    dinosaur = Dinosaur(GROUND_HEIGHT)
    bat = Batsymb(0, 115)
    obstacles = []
    lastObstacle = WIDTH
    ground_scroll = 0
    game_timer, score = 0, 0
    velocity = 300
    epsilon = 0.001  # Exploration rate

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
        if train and random.random() < epsilon:
            action = random.choice([0, 1])  # Exploration
        else:
            with torch.no_grad():
                action_prob = agent(state_tensor.to(device))
                action = torch.argmax(action_prob).item()  # Exploitation

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
                # gameover_sound.play()
                running = False
        
        lastObstacle -= velocity * delta_time

        # Update Dinosaur
        dinosaur.update(delta_time)
        dinosaur.draw(game_display)

        # Update Memory for RL
        next_state = get_game_state(dinosaur, obstacles, velocity)
        reward = calculate_reward(dinosaur, obstacles, running)
        memory.append((state, action, reward, next_state, not running))

        # Train Agent
        if train and len(memory) > 32:
            batch = random.sample(memory, 32)
            states, actions, rewards, next_states, dones = zip(*batch)


            states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
            actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)

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
    if score > 1000:
        print(f"Score exceeds 1000: {score}")
        running = False

    return score


if __name__ == "__main__":
    agent = DinoModel().to(device)
    agent.load_state_dict(torch.load("model.pth"))
    optimizer = optim.Adam(agent.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    memory = []

    scores = []  # List to store scores for each episode
    for episode in range(500):  # Train for 100 episodes
        score = run_game(agent, memory, optimizer, criterion, train=True)
        if score == 1000:
            break
        scores.append(score)
        print(f"Episode {episode + 1}: Score = {score}")

    # Save the trained model
    print(f"Max Score: {max(scores)}")
    torch.save(agent.state_dict(), "dino_model.pth")
    print("Training Complete. Model saved.")

