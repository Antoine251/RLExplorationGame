import numpy as np
import pygame

from environment.constants import MAP_SIZE
from environment.fish import Fish
from environment.map import TiledMap

pygame.init()
window = pygame.display.set_mode((MAP_SIZE * 2, MAP_SIZE))
pygame.display.set_caption("Brain project")

clock = pygame.time.Clock()


def main():
    map = TiledMap(window)
    fish = Fish(map.get_map(), window)
    run_brain = True
    i = 0
    while run_brain:
        i += 1
        print(i)
        # update background
        pygame.draw.rect(window, (100, 100, 100), (MAP_SIZE, MAP_SIZE / 2, MAP_SIZE, MAP_SIZE))  # type: ignore
        pygame.draw.rect(window, (150, 150, 150), (MAP_SIZE, -MAP_SIZE / 2, MAP_SIZE, MAP_SIZE))  # type: ignore

        # update map and pseudo-3D rendering
        map.draw()
        fish.draw()
        fish.cast_rays(map.map)

        # handle user input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            fish.orientation += 0.1
        if keys[pygame.K_RIGHT]:
            fish.orientation -= 0.1
        if keys[pygame.K_UP]:
            fish.position[0] += np.cos(fish.orientation) * 3
            fish.position[1] += np.sin(fish.orientation) * 3
            if fish.is_collision(map.get_map()):
                fish.position[0] -= np.cos(fish.orientation) * 3
                fish.position[1] -= np.sin(fish.orientation) * 3
        if keys[pygame.K_DOWN]:
            fish.position[0] -= np.cos(fish.orientation) * 3
            fish.position[1] -= np.sin(fish.orientation) * 3
            if fish.is_collision(map.get_map()):
                fish.position[0] += np.cos(fish.orientation) * 3
                fish.position[1] += np.sin(fish.orientation) * 3

        # update display
        pygame.display.flip()
        # set FPS
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                run_brain = False


if __name__ == "__main__":
    main()
