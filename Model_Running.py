import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dinosaur import Dinosaur  # Your Dinosaur class
from Arya_obstacle import Obstacle  # Your Obstacle class
from batsymbol import Batsymb  # Your Batsymbol class
from pygame import mixer

# Initialize pygame and mixer
pygame.init()
mixer.init()
clock = pygame.time.Clock()

# Game parameters
WIDTH, HEIGHT = 640, 480
GROUND_HEIGHT = HEIGHT - 100
MINGAP, MAXGAP = 200, 600
VELOCITY = 300
FPS = 60
MAXSIZE, MINSIZE = 40, 20

# Load assets
ground_image = pygame.image.load(r"ground.png")
ground_width = ground_image.get_width()
mixer.music.load("bgm.mp3")
mixer.music.set_volume(0.7)
achievement_sound = mixer.Sound("100.mp3")
gameover_sound = mixer.Sound("gameover.mp3")

# Fonts and colors
text_font = pygame.font.SysFont("Helvetica", 30)
white, black = (255, 255, 255), (0, 0, 0)

# Neural network for RL agent
class DinoModel(nn.Module):
    def __init__(self):
        super(DinoModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: [jump, duck]
        )

    def forward(self, x):
        return self.fc(x)

# Helper functions
def draw_text(text, font, color, x, y, game_display):
    img = font.render(text, True, color)
    game_display.blit(img, (x, y))

def get_game_state(dinosaur, obstacles, velocity):
    next_obstacle = next((obs for obs in obstacles if obs.x > dinosaur.x), None)
    if next_obstacle:
        state = [
            next_obstacle.x - dinosaur.x,
            next_obstacle.y,
            int(next_obstacle.is_high),
            velocity
        ]
    else:
        state = [WIDTH, GROUND_HEIGHT, 0, velocity]
    return np.array(state, dtype=np.float32)

# Game over screen
def game_over_screen(score, game_display):
    while True:
        game_display.fill(white)
        draw_text(f"Game Over. Score: {score}", text_font, (255, 0, 0), WIDTH // 2 - 100, HEIGHT // 2, game_display)
        # button(game_display, "Restart", WIDTH // 2 - 100, HEIGHT // 2 + 50, 100, 40, (100, 200, 100), (50, 150, 50), lambda: pygame.quit())
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

def button(game_display, text, x, y, width, height, inactive_color, active_color, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    color = active_color if x + width > mouse[0] > x and y + height > mouse[1] > y else inactive_color
    pygame.draw.rect(game_display, color, (x, y, width, height))
    btn_font = pygame.font.SysFont("Helvetica", 20)
    text_surf = btn_font.render(text, True, black)
    text_rect = text_surf.get_rect(center=(x + width // 2, y + height // 2))
    game_display.blit(text_surf, text_rect)
    if color == active_color and click[0] == 1 and action:
        action()

def run_game(agent, memory, optimizer, criterion, train=True):
    ground_scroll = 0  # Initialize ground_scroll here
    game_display = pygame.display.set_mode((WIDTH, HEIGHT))
    dinosaur = Dinosaur(GROUND_HEIGHT)
    obstacles = []
    score = 0
    game_timer = 0
    running = True

    epsilon = 0.1  # Exploration rate for epsilon-greedy policy

    while running:
        t = pygame.time.get_ticks()
        delta_time = clock.tick(FPS) / 1000.0
        game_timer += delta_time
        velocity = 300 + 0.01 * game_timer * 1000

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Ground scrolling
        ground_scroll -= velocity * delta_time
        if ground_scroll <= -ground_width:
            ground_scroll += ground_width
        game_display.fill(white)
        game_display.blit(ground_image, (ground_scroll, GROUND_HEIGHT - 50))
        game_display.blit(ground_image, (ground_scroll + ground_width, GROUND_HEIGHT - 50))

        # Draw score
        draw_text(f"Score: {int(game_timer * 10)}", text_font, (0, 255, 0), 50, 50, game_display)

        # Spawn obstacles
        if len(obstacles) == 0 or obstacles[-1].x < WIDTH - MINGAP:
            is_high = random.random() > 0.7
            size = random.randint(MINSIZE, MAXSIZE) if not is_high else 30
            obstacles.append(Obstacle(WIDTH, size, GROUND_HEIGHT, is_high))

        # Get current state
        state = get_game_state(dinosaur, obstacles, velocity)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Decide action
        if train and random.random() < epsilon:
            action = random.choice([0, 1])  # Exploration
        else:
            with torch.no_grad():
                action_prob = agent(state_tensor)
                action = torch.argmax(action_prob).item()  # Exploitation

        if action == 0:
            dinosaur.bigjump()
        elif action == 1:
            dinosaur.duck(True)
        else:
            dinosaur.duck(False)

        # Update and draw obstacles
        for obs in obstacles:
            obs.update(delta_time, velocity)
            obs.draw(game_display)
            dino_rect = pygame.Rect(dinosaur.x, GROUND_HEIGHT - dinosaur.y - dinosaur.height, dinosaur.width, dinosaur.height)
            obs_rect = pygame.Rect(obs.x, obs.y, obs.size, obs.size)
            if dino_rect.colliderect(obs_rect):
                mixer.music.stop()
                gameover_sound.play()
                game_over_screen(int(game_timer * 10), game_display)
                running = False

        # Update and draw dinosaur
        dinosaur.update(delta_time)
        dinosaur.draw(game_display)

        # Reward and memory update
        next_state = get_game_state(dinosaur, obstacles, velocity)
        reward = -1 if not running else 0.1
        memory.append((state, action, reward, next_state, not running))

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
        score += 1

    return int(game_timer * 10)  # Return the score when the game ends

if __name__ == "__main__":
    agent = DinoModel()
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    memory = []

    scores = []  # List to store scores for each episode
    for episode in range(100):  # Train for 100 episodes
        score = run_game(agent, memory, optimizer, criterion, train=True)
        scores.append(score)
        print(f"Episode {episode + 1}: Score = {score}")

    # Save the trained model
    torch.save(agent.state_dict(), "dino_model.pth")
    print("Training Complete. Model saved.")

