import pygame

import pygame

pygame.init()
gameDisplay = pygame.display.set_mode((800,600))
done =False

while not done:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            done=True
    pygame.display.flip()
# pygame.display.set_caption('A bit Racey')
pygame.rect