import pygame
from dinosaur import Dinosaur #import the class Dinosaur from the file ’dinosaur’
from obstacle_new import Obstacle
from batsymbol import Batsymb
import csv
import time
import numpy as np

pygame.init() #this ‘starts up’ pygame
from pygame import mixer 

# Starting the mixer 
mixer.init() 
  
# Loading the song 
#mixer.music.load("bgm.mp3") 

Bat =  Batsymb(0, 115) 
# Setting the volume 
mixer.music.set_volume(0.7) 
  
# Start playing the song 
#mixer.music.play(loops = -1) 
  
size = width,height = 640, 480#creates tuple called size with width 400  and height 230 
gameDisplay= pygame.display.set_mode(size) #creates screen
xPos = 0
yPos = 0
black = 0,0,0
GROUND_HEIGHT = height-100
collision = False  # Track collision state
collision_animation_complete = False
WIDTH, HEIGHT = 640, 480


dinosaur = Dinosaur(GROUND_HEIGHT)

lastFrame = pygame.time.get_ticks() #get ticks returns current time in milliseconds

import random
MINGAP = 200

MAXGAP = 600
MAXSIZE = 40
MINSIZE = 20
obstacles = []
lastObstacle = width
text_font = pygame.font.SysFont("Helvetica", 30)

obstaclesize = 20

ground_image = pygame.image.load(r"ground.png")  # Load the ground texture
ground_width = ground_image.get_width()        # Get the width of the texture
ground_scroll = 0

def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    gameDisplay.blit(img, (x, y))

white = 255,255,255

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

training_data = []  # List to store game states and actions
prev_action = -1

try:
    while True:  # Game loop
        t = pygame.time.get_ticks()  # Get current time
        deltaTime = (t - lastFrame) / 1000.0  # Find difference in time and then convert it to seconds
        lastFrame = t  # Set lastFrame as the current time for the next frame.
        VELOCITY = 300 + 0.01*t

        action = 2  
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()

            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # Jump action handling
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Jump when space is pressed
                    dinosaur.bigjump()
                    action = 0  # Jump action

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:  # Smol jump
                    dinosaur.smoljump()
                    action = 0  # Jump action

            # Duck action handling (ensure it doesn't overwrite the jump action)
            if keys[pygame.K_DOWN]:
                dinosaur.duck(True)  # Duck while the down key is held
                if action == 2:  # Don't overwrite jump action if already jumping
                    action = 1  # Duck action
            else:
                dinosaur.duck(False)

        # Log game state and action
        game_state = get_game_state(dinosaur, obstacles, VELOCITY)
        if action != 2 or t%100 == 0:  # Log data only when action is not 'do nothing'
            training_data.append([t] + game_state.tolist() + [action])
            

        gameDisplay.fill(white)  # Clear the screen
        ground_scroll -= VELOCITY * deltaTime
        if ground_scroll <= -ground_width:  # Reset scroll position when it goes off-screen
            ground_scroll += ground_width
        gameDisplay.blit(ground_image, (ground_scroll, 300))            # First segment
        gameDisplay.blit(ground_image, (ground_scroll + ground_width, 300))
        # Draw Score
        draw_text(f"Score: {t // 100}", text_font, (0, 255, 0), 100, 50)

        dinosaur.update(deltaTime)
        dinosaur.draw(gameDisplay)
        Bat.update(deltaTime)
        Bat.draw(gameDisplay)
        if len(obstacles) == 0 or obstacles[-1].x < width - MINGAP:
            ahaha = random.random()
            is_high = ahaha > 0.7
            obstacle_size = random.randint(MINSIZE, MAXSIZE) if not is_high else 30
            obstacles.append(Obstacle(lastObstacle, obstacle_size, GROUND_HEIGHT, is_high))
            lastObstacle += MINGAP + (MAXGAP - MINGAP) * random.random() + 0.01*t

        # Check for collisions and update obstacles
        for obs in obstacles:
            if not collision:
                obs.update(deltaTime, VELOCITY)
                obs.draw(gameDisplay)

            # Define dinosaur and obstacle rectangles
                dino_rect = pygame.Rect(
                dinosaur.x, dinosaur.surfaceHeight - dinosaur.y - dinosaur.height, dinosaur.width, dinosaur.height
            )
            obs_rect = pygame.Rect(
                obs.x, obs.y, obs.size, obs.size
            )

            # Check for collision
            if dino_rect.colliderect(obs_rect):
                draw_text("Game Over", text_font, (255, 0, 0), width // 2 - 100, height // 2)
                pygame.display.update()
                mixer.music.load("gameover.mp3")    
                # Setting the volume 
                mixer.music.set_volume(0.7) 
                # Start playing the song 
                mixer.music.play() 
                pygame.time.wait(3000)  # Wait for 2 seconds
                pygame.quit()
                quit()

            dinosaur.draw(gameDisplay)

            if obs.checkOver():  # Reset obstacle position if it goes off-screen
                obstacles.remove(obs)

        lastObstacle -= VELOCITY * deltaTime

        # pygame.draw.rect(gameDisplay, black, [0, GROUND_HEIGHT, width, height - GROUND_HEIGHT])
        pygame.display.update()  # Updates the screen
        pass
finally:
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Format: YYYYMMDD-HHMMSS
    filename = f"Train_Data/{timestamp}.csv"  # Example: dino_training_data_20241118-143500.csv

    # Create and write to the file
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time","Obstacle Distance","Obstacle Height", "Obstacle Size", "Velocity", "Dino Jumping", "Dino Ducking", "Action"])
        writer.writerows(training_data)
    print("Training data saved!")