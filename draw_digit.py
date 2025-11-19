import pygame
import numpy as np
from PIL import Image

WIDTH, HEIGHT = 280, 280
WHITE =(255,255,255)
BLACK=(0,0,0)
RADIUS=8

pygame.init()
screen=pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Draw digit")
screen.fill(WHITE)

running=True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if pygame.mouse.get_pressed()[0]:
            pos=pygame.mouse.get_pos()
            pygame.draw.circle(screen,BLACK,pos,RADIUS)
        if pygame.mouse.get_pressed()[2]:
            pos=pygame.mouse.get_pos()
            pygame.draw.circle(screen,WHITE,pos,RADIUS)
        if event.type== pygame.KEYDOWN and event.key == pygame.K_RETURN:
            pygame.image.save(screen,"my_digit.png")
            running=False
    pygame.display.flip()
pygame.quit()

img=Image.open("my_digit.png").convert("L")
img=img.resize((28,28),Image.ANTIALIAS)
img.save("my_digit.png")
