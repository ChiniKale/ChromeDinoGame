import gym
from gym import spaces
import numpy as np
import pygame
from dinosaur import Dinosaur
from obstacle import Obstacle
from batsymbol import Batsymb
from pygame import mixer
import random

# Initialize Pygame and the mixer
pygame.init()
mixer.init()

class DinoGameEnv(gym.Env):
    def __init__(self):
        super(DinoGameEnv, self).__init__()
        self.size = self.width, self.height = 640, 480
        self.GROUND_HEIGHT = self.height - 100
        self.gameDisplay = pygame.display.set_mode(self.size)

        # Initialize game elements
        self.dinosaur = Dinosaur(self.GROUND_HEIGHT)
        self.obstacles = []
        self.Bat = Batsymb(0, 115)
        self.lastObstacle = self.width
        self.VELOCITY = 300
        self.game_timer = 0
        self.done = False

        # Load assets
        self.ground_image = pygame.image.load(r"ground.png")
        self.ground_width = self.ground_image.get_width()
        self.ground_scroll = 0
        mixer.music.load("bgm.mp3")
        mixer.music.play(loops=-1)

        # Define action space (0: Do nothing, 1: Jump, 2: Duck)
        self.action_space = spaces.Discrete(3)

        # Observation space: [Obstacle Distance, Obstacle Size, Is High, Dino Y, Velocity]
        low = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([self.width, 50, 1, self.height, 1000], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        """Reset the game environment."""
        self.dinosaur = Dinosaur(self.GROUND_HEIGHT)
        self.obstacles = []
        self.lastObstacle = self.width
        self.ground_scroll = 0
        self.game_timer = 0
        self.done = False
        mixer.music.play(loops=-1)
        return self.get_state()

    def step(self, action):
        """Advance the game state by applying an action."""
        deltaTime = 1 / 60  # Fixed time step (60 FPS)
        self.game_timer += deltaTime
        reward = 1  # Reward for surviving one frame
        self.VELOCITY = 300 + 0.01 * self.game_timer * 1000

        # Apply action
        if action == 1:  # Jump
            self.dinosaur.bigjump()
        elif action == 2:  # Duck
            self.dinosaur.duck(True)
        else:  # Do nothing
            self.dinosaur.duck(False)

        # Update game elements
        self.dinosaur.update(deltaTime)
        self.ground_scroll -= self.VELOCITY * deltaTime
        if self.ground_scroll <= -self.ground_width:
            self.ground_scroll += self.ground_width

        # Spawn and update obstacles
        if len(self.obstacles) == 0 or self.obstacles[-1].x < self.width - 200:
            is_high = random.random() > 0.7
            obstacle_size = random.randint(20, 40) if not is_high else 30
            self.obstacles.append(Obstacle(self.lastObstacle, obstacle_size, self.GROUND_HEIGHT, is_high))
            self.lastObstacle += 200 + (600 - 200) * random.random()

        for obs in self.obstacles:
            obs.update(deltaTime, self.VELOCITY)
            if obs.checkOver():
                self.obstacles.remove(obs)

        # Check for collisions
        for obs in self.obstacles:
            if self.check_collision(self.dinosaur, obs):
                reward = -10
                self.done = True
                mixer.music.load("gameover.mp3")
                mixer.music.play()
                break

        # Prepare the next state
        state = self.get_state()
        return state, reward, self.done, {}

    def get_state(self):
        """Get the current game state."""
        next_obstacle = self.obstacles[0] if self.obstacles else None
        obstacle_distance = next_obstacle.x - self.dinosaur.x if next_obstacle else float('inf')
        obstacle_size = next_obstacle.size if next_obstacle else 0
        is_high = next_obstacle.y if next_obstacle else 0
        return np.array([obstacle_distance, obstacle_size, is_high, self.dinosaur.y, self.VELOCITY], dtype=np.float32)

    def check_collision(self, dinosaur, obstacle):
        """Check for collision."""
        dino_rect = pygame.Rect(
            dinosaur.x, dinosaur.surfaceHeight - dinosaur.y - dinosaur.height, dinosaur.width, dinosaur.height
        )
        obs_rect = pygame.Rect(obstacle.x, obstacle.y, obstacle.size, obstacle.size)
        return dino_rect.colliderect(obs_rect)

    def render(self, mode="human"):
        """Render the game."""
        self.gameDisplay.fill((255, 255, 255))
        self.gameDisplay.blit(self.ground_image, (self.ground_scroll, 300))
        self.gameDisplay.blit(self.ground_image, (self.ground_scroll + self.ground_width, 300))
        self.dinosaur.draw(self.gameDisplay)
        for obs in self.obstacles:
            obs.draw(self.gameDisplay)
        pygame.display.update()

    def close(self):
        """Close the environment."""
        pygame.quit()
