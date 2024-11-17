import pygame
from dinosaur import Dinosaur #import the class Dinosaur from the file ’dinosaur’
from obstacle import Obstacle

pygame.init() #this ‘starts up’ pygame
from pygame import mixer 
  
# Starting the mixer 
mixer.init() 
  
# Loading the song 
mixer.music.load("bgm.mp3") 
  
# Setting the volume 
mixer.music.set_volume(0.7) 
  
# Start playing the song 
mixer.music.play(loops = -1) 
  
size = width,height = 640, 480#creates tuple called size with width 400  and height 230 
gameDisplay= pygame.display.set_mode(size) #creates screen
xPos = 0
yPos = 0
black = 0,0,0
GROUND_HEIGHT = height-100 
collision = False  # Track collision state
collision_animation_complete = False


dinosaur = Dinosaur(GROUND_HEIGHT)

lastFrame = pygame.time.get_ticks() #get ticks returns current time in milliseconds

import random
MINGAP = 200
VELOCITY = 300
MAXGAP = 600
MAXSIZE = 40
MINSIZE = 20
obstacles = []
lastObstacle = width
text_font = pygame.font.SysFont("Helvetica", 30)

obstaclesize = 20


def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    gameDisplay.blit(img, (x, y))

white = 255,255,255
while True:  # Game loop
    t = pygame.time.get_ticks()  # Get current time
    deltaTime = (t - lastFrame) / 1000.0  # Find difference in time and then convert it to seconds
    lastFrame = t  # Set lastFrame as the current time for the next frame.

    for event in pygame.event.get():  # Check for events
        keys = pygame.key.get_pressed()
        if event.type == pygame.QUIT:
            pygame.quit()  # Quit
            quit()
        if event.type == pygame.KEYDOWN:  # If user uses the keyboard
            if event.key == pygame.K_SPACE:  # If that key is space
                dinosaur.jump()  # Make dinosaur jump
        if keys[pygame.K_DOWN]:
            dinosaur.duck(True)  # Duck while the down key is held
        else:
            dinosaur.duck(False)          

    gameDisplay.fill(white)  # Clear the screen

    # Draw Score
    draw_text(f"Score: {t // 1000}", text_font, (0, 255, 0), 100, 150)

    dinosaur.update(deltaTime)
    dinosaur.draw(gameDisplay)
    if len(obstacles) == 0 or obstacles[-1].x < width - MINGAP:
        is_high = random.random() > 0.7  # 30% chance to be ground obstacle, 70% sky
        obstacle_size = random.randint(MINSIZE, MAXSIZE) if not is_high else 30
        obstacles.append(Obstacle(lastObstacle, obstacle_size, GROUND_HEIGHT, is_high))
        lastObstacle += MINGAP + (MAXGAP - MINGAP) * random.random()

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

    pygame.draw.rect(gameDisplay, black, [0, GROUND_HEIGHT, width, height - GROUND_HEIGHT])
    pygame.display.update()  # Updates the screen
