import pygame
import random
import torch
import torch.nn as nn
import numpy as np
from pygame import mixer
from dinosaur import Dinosaur
from obstacle import Obstacle
from batsymbol import Batsymb

# Initialize Pygame and Mixer
pygame.init()
mixer.init()
clock = pygame.time.Clock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Constants
WIDTH, HEIGHT = 640, 480
GROUND_HEIGHT = HEIGHT - 100
FPS = 60
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
MINGAP = 100
MAXGAP = 300
MAXSIZE = 40
MINSIZE = 20
POPULATION_SIZE = 10
GENERATIONS = 200
MUTATION_RATE = 0.2

# Load Assets with Error Handling
try:
    ground_image = pygame.image.load("ground.png")
    mixer.music.load("bgm.mp3")
    achievement_sound = mixer.Sound("100.mp3")
except pygame.error as e:
    print(f"Error loading assets: {e}")
    quit()

ground_width = ground_image.get_width()


class DinoModel(nn.Module):
    """Neural network for the dinosaur agent."""
    def __init__(self):
        super(DinoModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Outputs: [jump, duck, do nothing]
        )

    def forward(self, x):
        return self.fc(x)


def draw_text(text, font, color, x, y, surface):
    """Draw text on the screen."""
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect(topleft=(x, y))
    surface.blit(text_obj, text_rect)


def get_game_state(dino, obstacles, velocity):
    """Extract the current game state for the agent."""
    next_obstacles = [obs for obs in obstacles if obs.x > dino.x]
    state = [WIDTH, 0, 0, velocity, int(dino.is_jumping), int(dino.is_ducking)]

    if next_obstacles:
        next_obs = next_obstacles[0]
        state[:3] = [next_obs.x - dino.x, next_obs.y, next_obs.size]

    return np.array(state, dtype=np.float32)


def update_and_draw_obstacles(obstacles, delta_time, velocity, surface):
    """Update and draw obstacles."""
    for obs in obstacles:
        obs.update(delta_time, velocity)
        obs.draw(surface)


def crossover(parent1, parent2):
    """Cross two neural networks to produce a child network."""
    child = DinoModel()
    with torch.no_grad():
        for cp, p1, p2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            mask = torch.rand_like(p1) > 0.5
            cp.copy_(p1 * mask + p2 * ~mask)
    return child


def mutate(agent, rate=0.1):
    """Apply random mutations to the agent's parameters."""
    with torch.no_grad():
        for param in agent.parameters():
            mask = torch.rand_like(param) < rate
            param.add_(mask * torch.randn_like(param) * 0.1)
    return agent


def run_generation(population, num_dinos=10):
    """Run one generation of the game."""
    game_display = pygame.display.set_mode((WIDTH, HEIGHT))
    dinosaurs = [Dinosaur(GROUND_HEIGHT) for _ in range(num_dinos)]
    agents = [agent.to(device) for agent in population]
    scores = [0] * num_dinos
    obstacles, last_obstacle = [], WIDTH
    ground_scroll = 0
    game_timer, velocity = 0, 300
    running = [True] * num_dinos

    while any(running):
        delta_time = clock.tick(FPS) / 1000.0
        game_timer += delta_time
        velocity = 300 + 0.01 * game_timer * 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Update ground
        ground_scroll = (ground_scroll - velocity * delta_time) % ground_width
        game_display.fill(WHITE)
        game_display.blit(ground_image, (ground_scroll, 300))
        game_display.blit(ground_image, (ground_scroll + ground_width, 300))

        # Display scores
        score = max(scores)
        draw_text(f"Score: {score}", pygame.font.SysFont("Helvetica", 30), (0, 255, 0), 50, 50, game_display)

        # Spawn and update obstacles
        if not obstacles or obstacles[-1].x < WIDTH - MINGAP:
            is_high = random.random() > 0.4
            obs_size = random.randint(MINSIZE, MAXSIZE) if not is_high else 30
            obstacles.append(Obstacle(last_obstacle, obs_size, GROUND_HEIGHT, is_high))
            last_obstacle += MINGAP + (MAXGAP - MINGAP) * random.random() + 0.01 * game_timer * 1000

        update_and_draw_obstacles(obstacles, delta_time, velocity, game_display)

        # Update dinosaurs
        for i, dino in enumerate(dinosaurs):
            if not running[i]:
                continue

            state = get_game_state(dino, obstacles, velocity)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action = torch.argmax(agents[i](state_tensor)).item()

            if action == 0:
                dino.bigjump()
            elif action == 1:
                dino.duck(True)
            else:
                dino.duck(False)

            dino.update(delta_time)
            dino.draw(game_display)

            for obs in obstacles:
                if pygame.Rect(dino.x, GROUND_HEIGHT - dino.y - dino.height, dino.width, dino.height).colliderect(
                    pygame.Rect(obs.x, obs.y, obs.size, obs.size)
                ):
                    running[i] = False

        pygame.display.update()

    return scores


def evolve_population(population, scores, num_parents=2, rate=0.1):
    """Evolve the population based on fitness scores."""
    sorted_indices = np.argsort(scores)[::-1]
    parents = [population[i] for i in sorted_indices[:num_parents]]
    next_population = []

    while len(next_population) < len(population):
        p1, p2 = random.sample(parents, 2)
        child = mutate(crossover(p1, p2), rate)
        next_population.append(child)

    return next_population


if __name__ == "__main__":
    # Initialize population
    population = [DinoModel().to(device) for _ in range(POPULATION_SIZE)]

    for gen in range(GENERATIONS):
        print(f"Generation {gen + 1}")
        scores = run_generation(population, num_dinos=POPULATION_SIZE)
        print(f"Max Score: {max(scores)}, Avg Score: {sum(scores) / len(scores)}")
        population = evolve_population(population, scores, num_parents=2, rate=MUTATION_RATE)

    # Save the best model
    torch.save(population[0].state_dict(), "dino_genetic_model.pth")
    print("Training Complete. Model saved.")
