import pygame
from dinosaur import Dinosaur #import the class Dinosaur from the file ’dinosaur’
from obstacle import Obstacle

pygame.init() #this ‘starts up’ pygame

size = width,height = 640, 480#creates tuple called size with width 400  and height 230 
gameDisplay= pygame.display.set_mode(size) #creates screen
xPos = 0
yPos = 0
black = 0,0,0
GROUND_HEIGHT = height-100 


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
for i in range(4):
    lastObstacle += MINGAP+(MAXGAP-MINGAP)*random.random() #Make distance between rocks random
    obstacle_size = random.randint(MINSIZE, MAXSIZE)
    is_high = random.random() > 0.7
    obstacles.append(Obstacle(lastObstacle, obstacle_size, GROUND_HEIGHT, is_high))


def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    gameDisplay.blit(img, (x, y))

white = 255,255,255
while True:  # Game loop
    t = pygame.time.get_ticks()  # Get current time
    deltaTime = (t - lastFrame) / 1000.0  # Find difference in time and then convert it to seconds
    lastFrame = t  # Set lastFrame as the current time for the next frame.

    for event in pygame.event.get():  # Check for events
        if event.type == pygame.QUIT:
            pygame.quit()  # Quit
            quit()
        if event.type == pygame.KEYDOWN:  # If user uses the keyboard
            if event.key == pygame.K_SPACE:  # If that key is space
                dinosaur.jump()  # Make dinosaur jump

    gameDisplay.fill(black)  # Clear the screen

    # Draw Score
    draw_text(f"Score: {t // 1000}", text_font, (0, 255, 0), 100, 150)

    dinosaur.update(deltaTime)
    dinosaur.draw(gameDisplay)

    # Check for collisions and update obstacles
    for obs in obstacles:
        obs.update(deltaTime, VELOCITY)
        obs.draw(gameDisplay)

        # Define dinosaur and obstacle rectangles
        dino_rect = pygame.Rect(
            dinosaur.x, dinosaur.surfaceHeight - dinosaur.y - dinosaur.height, dinosaur.width, dinosaur.height
        )
        obs_rect = pygame.Rect(
            obs.x, obs.GroundHeight - obs.size, obs.size, obs.size
        )

        # Check for collision
        if dino_rect.colliderect(obs_rect):
            draw_text("Game Over", text_font, (255, 0, 0), width // 2 - 100, height // 2)
            pygame.display.update()
            pygame.time.wait(2000)  # Wait for 2 seconds
            pygame.quit()
            quit()

        if obs.checkOver():  # Reset obstacle position if it goes off-screen
            lastObstacle += MINGAP + (MAXGAP - MINGAP) * random.random()
            obs.x = lastObstacle

    lastObstacle -= VELOCITY * deltaTime

    pygame.draw.rect(gameDisplay, white, [0, GROUND_HEIGHT, width, height - GROUND_HEIGHT])
    pygame.display.update()  # Updates the screen
