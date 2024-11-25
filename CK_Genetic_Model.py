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

class DinoModel(nn.Module):
    def __init__(self):
        super(DinoModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 256),  # Increased input size and more neurons
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Output: [jump, duck, do nothing]
        )
        # Softmax will be applied outside this model during action selection.

    def forward(self, x):
        return self.fc(x)

# DinoModel Neural Network
# class DinoModel(nn.Module):
#     def __init__(self):
#         super(DinoModel, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(6, 128),  # Increased size for initial layer
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 3)  # Output: [jump, duck,do nothing]
#         )

#     def forward(self, x):
#         return self.fc(x)

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
            next_obstacle.y,  # Vertical distance
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

def no_obstacle_nearby(dino, obstacles):
    """Check if no obstacles are close to the dinosaur."""
    for obs in obstacles:
        # Check if obstacle is far away
        if obs.x > dino.x + dino.width + 50:  # Safe distance threshold
            continue
        return False  # Obstacle is near
    return True

def is_ducking_necessary(dino, obstacles):
    """Check if ducking is necessary for high obstacles."""
    for obs in obstacles:
        # Check if obstacle is near and is high
        if obs.x < dino.x + dino.width and obs.is_high:
            return True
    return False

def crossover(parent1, parent2):
    """Cross two neural networks to produce a child network."""
    child = DinoModel()
    with torch.no_grad():
        for child_param, param1, param2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            # Randomly select weights from each parent
            mask = torch.rand_like(param1) > 0.5
            child_param.copy_(param1 * mask + param2 * ~mask)
    return child

def mutate(agent, mutation_rate=0.1):
    """Apply random mutations to the agent's parameters."""
    with torch.no_grad():
        for param in agent.parameters():
            mutation_mask = torch.rand_like(param) < mutation_rate
            param.add_(mutation_mask * torch.randn_like(param) * 0.1)  # Small random mutations
    return agent

def is_jumping_necessary(dino, obstacles):
    """Check if jumping is necessary based on obstacles."""
    for obs in obstacles:
        # Check if obstacle is near and high enough to need a jump
        if obs.x < dino.x + dino.width and obs.y > GROUND_HEIGHT - 100:
            return True  # Jump is needed to avoid the obstacle
    return False  # No need to jump


def run_generation(population, num_dinos=10):
    """Run one generation of the game."""
    game_display = pygame.display.set_mode((WIDTH, HEIGHT))
    dinosaurs = [Dinosaur(GROUND_HEIGHT) for _ in range(num_dinos)]
    agents = [agent.to(device) for agent in population]
    bat = Batsymb(0, 115)
    scores = [0] * num_dinos
    obstacles = []
    lastObstacle = WIDTH
    ground_scroll = 0
    game_timer, velocity = 0, 300
    running = [True] * num_dinos

    excessive_jump_penalty = -500  # Set a penalty for jumping too often

    while any(running):
        delta_time = clock.tick(FPS) / 1000.0
        game_timer += delta_time
        velocity = 300 + 0.01 * game_timer * 1000

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
        for i, dino in enumerate(dinosaurs):
            if running[i]:
                scores[i] = int(game_timer * 10)
        score = max(scores)
        draw_text(f"Score: {scores}", pygame.font.SysFont("Helvetica", 30), (0, 255, 0), 50, 50, game_display)        

        # Spawn Obstacles
        if len(obstacles) == 0 or obstacles[-1].x < WIDTH - MINGAP:
            is_high = random.random() > 0.7
            obstacle_size = random.randint(MINSIZE, MAXSIZE) if not is_high else 30
            obstacles.append(Obstacle(lastObstacle, obstacle_size, GROUND_HEIGHT, is_high))
            lastObstacle += MINGAP + (MAXGAP - MINGAP) * random.random() + 0.01 * game_timer * 500

        # Update Bat
        bat.update(delta_time)
        bat.draw(game_display)

        # Update and Draw Obstacles
        for obs in obstacles:
            obs.update(delta_time, velocity)
            obs.draw(game_display)

        # Update Dinosaurs
        for i, dino in enumerate(dinosaurs):
            if not running[i]:
                continue

            # Get Current State
            state = get_game_state(dino, obstacles, velocity)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            # Decide Action
            with torch.no_grad():
                action_prob = agents[i](state_tensor)
                action = torch.argmax(action_prob).item()

            # Perform Action
            if action == 0:  # Jump action
                # Check if jumping is necessary (e.g., there's an obstacle in front)
                if not is_jumping_necessary(dino, obstacles):
                    # Penalize excessive jumping
                    scores[i] += excessive_jump_penalty
                else:
                    # Reward for jumping to avoid an obstacle
                    scores[i] += 300
                dino.bigjump()

            elif action == 1:  # Duck action
                # Check if ducking is necessary for high obstacles
                if is_ducking_necessary(dino, obstacles):
                    scores[i] += 300  # Reward for successfully ducking under a high obstacle
                else:
                    scores[i] += excessive_jump_penalty  # Penalty for unnecessary ducking
                dino.duck(True)

            else:  # Do nothing
                # Check if doing nothing is the right choice (e.g., no obstacle nearby)
                if no_obstacle_nearby(dino, obstacles):
                    scores[i] += 300  # Reward for doing nothing when safe
                dino.duck(False)


            # Update and Check for Collisions
            dino.update(delta_time)
            dino.draw(game_display)

            for obs in obstacles:
                dino_rect = pygame.Rect(dino.x, GROUND_HEIGHT - dino.y - dino.height, dino.width, dino.height)
                obs_rect = pygame.Rect(obs.x, obs.y, obs.size, obs.size)
                if dino_rect.colliderect(obs_rect):
                    running[i] = False  # Dino is out

        pygame.display.update()

    return scores


def evolve_population(population, scores, num_parents=2, mutation_rate=0.1):
    """Evolve the population based on fitness scores."""
    # Sort by scores (fitness)
    sorted_indices = np.argsort(scores)[::-1]
    top_agents = [population[i] for i in sorted_indices[:num_parents]]

    # Create next generation
    next_population = []
    while len(next_population) < len(population):
        parent1, parent2 = random.sample(top_agents, 2)
        child = crossover(parent1, parent2)
        child = mutate(child, mutation_rate)
        next_population.append(child)

    return next_population



if __name__ == "__main__":
    # Initialize Population
    population_size = 10
    generations = 50
    mutation_rate = 0.1
    population = [DinoModel().to(device) for _ in range(population_size)]

    for generation in range(generations):
        print(f"Generation {generation + 1}")
        scores = run_generation(population, num_dinos=population_size)

        # Print Stats
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        print(f"Max Score: {max_score}, Avg Score: {avg_score}")

        # Evolve
        population = evolve_population(population, scores, num_parents=2, mutation_rate=mutation_rate)

    # Save the best-performing model
    best_agent = population[0]
    torch.save(best_agent.state_dict(), "dino_genetic_model.pth")
    print("Training Complete. Model saved.")

# if __name__ == "__main__":
#     # Initialize Population
#     population_size = 10
#     generations = 20
#     mutation_rate = 0.5
    
#     # Load the pre-trained model
#     agent = DinoModel().to(device)
#     agent.load_state_dict(torch.load("dino_genetic_model1.pth"))
#     agent.eval()  # Set to evaluation mode
    
#     # Initialize the population with copies of the pre-trained agent
#     population = [DinoModel().to(device) for _ in range(population_size)]
#     for i in range(population_size):
#         population[i].load_state_dict(agent.state_dict())  # Copy weights from the pre-trained agent

#     for generation in range(generations):
#         print(f"Generation {generation + 1}")
#         scores = run_generation(population, num_dinos=population_size)

#         # Print Stats
#         max_score = max(scores)
#         avg_score = sum(scores) / len(scores)
#         print(f"Max Score: {max_score}, Avg Score: {avg_score}")

#         # Evolve the population
#         population = evolve_population(population, scores, num_parents=2, mutation_rate=mutation_rate)

#     # Save the best-performing model
#     best_agent = population[0]
#     torch.save(best_agent.state_dict(), "dino_genetic_model2.pth")
#     print("Training Complete. Model saved.")

