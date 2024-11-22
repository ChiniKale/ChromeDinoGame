import pygame
from dinosaur import Dinosaur #import the class Dinosaur from the file ’dinosaur’
from obstacle import Obstacle
from batsymbol import Batsymb

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
game_timer = 0
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

def reset_game_and_exit_gameover():
    global game_over
    reset_game()
    game_over = False

def button(text, x, y, width, height, inactive_color, active_color, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x + width > mouse[0] > x and y + height > mouse[1] > y:
        pygame.draw.rect(gameDisplay, active_color, (x, y, width, height))
        if click[0] == 1 and action is not None:
            action()
    else:
        pygame.draw.rect(gameDisplay, inactive_color, (x, y, width, height))
    # Render button text
    btn_font = pygame.font.SysFont("Helvetica", 20)
    text_surf = btn_font.render(text, True, white)
    text_rect = text_surf.get_rect(center=(x + width // 2, y + height // 2))
    gameDisplay.blit(text_surf, text_rect)


def reset_game():
    global game_timer, lastFrame, dinosaur, obstacles, lastObstacle, ground_scroll
    game_timer = 0  # Reset game timer
    lastFrame = pygame.time.get_ticks()
    dinosaur = Dinosaur(GROUND_HEIGHT)
    obstacles = []
    lastObstacle = width
    ground_scroll = 0  # Reset ground scrolling position
    mixer.music.load("bgm.mp3")
    mixer.music.play(loops=-1)


white = 255,255,255
while True:  # Game loop
    t = pygame.time.get_ticks()
    deltaTime = (t - lastFrame) / 1000.0
    lastFrame = t

    # Increment game timer using delta time
    game_timer += deltaTime
    VELOCITY = 300 + 0.01 * game_timer * 1000  # Adjust velocity based on game timer

    for event in pygame.event.get():
        keys = pygame.key.get_pressed()
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                dinosaur.bigjump()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                dinosaur.smoljump()
        if keys[pygame.K_DOWN]:
            dinosaur.duck(True)
        else:
            dinosaur.duck(False)

    gameDisplay.fill(white)
    ground_scroll -= VELOCITY * deltaTime
    if ground_scroll <= -ground_width:
        ground_scroll += ground_width

    gameDisplay.blit(ground_image, (ground_scroll, 300))
    gameDisplay.blit(ground_image, (ground_scroll + ground_width, 300))

    # Draw Score
    draw_text(f"Score: {int(game_timer)}", text_font, (0, 255, 0), 100, 50)

    dinosaur.update(deltaTime)
    dinosaur.draw(gameDisplay)
    Bat.update(deltaTime)
    Bat.draw(gameDisplay)

    if len(obstacles) == 0 or obstacles[-1].x < width - MINGAP:
        ahaha = random.random()
        is_high = ahaha > 0.7
        obstacle_size = random.randint(MINSIZE, MAXSIZE) if not is_high else 30
        obstacles.append(Obstacle(lastObstacle, obstacle_size, GROUND_HEIGHT, is_high))
        lastObstacle += MINGAP + (MAXGAP - MINGAP) * random.random() + 0.01 * game_timer * 1000

    # Check for collisions and update obstacles
    for obs in obstacles:
        obs.update(deltaTime, VELOCITY)
        obs.draw(gameDisplay)

        dino_rect = pygame.Rect(
            dinosaur.x, dinosaur.surfaceHeight - dinosaur.y - dinosaur.height, dinosaur.width, dinosaur.height
        )
        obs_rect = pygame.Rect(obs.x, obs.y, obs.size, obs.size)

        if dino_rect.colliderect(obs_rect):
            mixer.music.load("gameover.mp3")
            mixer.music.set_volume(0.7)
            mixer.music.play()

            game_over = True
            while game_over:
                gameDisplay.fill(white)
                draw_text("Game Over", text_font, (255, 0, 0), width // 2 - 100, height // 2)
                button("Restart", width // 2 - 50, height // 2 + 50, 100, 40, (100, 200, 100), (50, 150, 50), lambda: reset_game_and_exit_gameover())

                pygame.display.update()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

    lastObstacle -= VELOCITY * deltaTime
    pygame.display.update()
