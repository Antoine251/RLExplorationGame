from typing import Union

from environment.fish import Fish
from environment.map import TiledMap
import pygame
from environment.constants import MAP_SIZE
import numpy as np


class Environment:
    def __init__(self) -> None:
        pygame.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((MAP_SIZE * 2, MAP_SIZE))

        pygame.display.set_caption("Brain project")
        self.reward_font = pygame.font.Font(None, 25)
        self.mean_reward_font = pygame.font.Font(None, 25)

        self.clock = pygame.time.Clock()

        self.map = TiledMap(self.window)
        self.fish = Fish(self.map.get_map(), self.window)
        self.fish2 = Fish(self.map.get_map(), self.window)
        self.counter = 0
        self.cumulative_reward = 0

    def reset(self):
        self.map = TiledMap(self.window)
        self.fish = Fish(self.map.get_map(), self.window)
        self.fish2 = Fish(self.map.get_map(), self.window)
        self.clock.tick(0)
        self.counter = 0
        self.cumulative_reward = 0

        self.map.draw()
        self.fish.draw()
        self.fish.cast_rays(self.map.map)

        return self.fish.vision / 255

    def step(
        self, action: Union[pygame.key.ScancodeWrapper, list], last_rewards: list = 0  # type:ignore
    ):
        reward = 0
        for _ in range(6):
            self.counter += 1
            pygame.draw.rect(
                self.window,
                (100, 100, 100),
                (MAP_SIZE, MAP_SIZE / 2, MAP_SIZE, MAP_SIZE),  # type:ignore
            )
            pygame.draw.rect(
                self.window,
                (150, 150, 150),
                (MAP_SIZE, -MAP_SIZE / 2, MAP_SIZE, MAP_SIZE),  # type:ignore
            )

            # handle user input
            if isinstance(action, list):
                left_index = 0
                right_index = 2
                up_index = 1
            else:
                left_index = pygame.K_LEFT
                right_index = pygame.K_RIGHT
                up_index = pygame.K_UP

            if action[left_index]:
                self.fish.orientation += 0.1
            if action[right_index]:
                self.fish.orientation -= 0.1
            if action[up_index]:
                reward += 0.2
                self.fish.position[0] += np.cos(self.fish.orientation) * 3
                self.fish.position[1] += np.sin(self.fish.orientation) * 3
                if self.fish.is_collision(self.map.get_map()):
                    return self.fish.vision / 255, -10, True

            # if action[pygame.K_DOWN]:
            #     self.fish.position[0] -= np.cos(self.fish.orientation)*3
            #     self.fish.position[1] -= np.sin(self.fish.orientation)*3
            #     if self.fish.is_collision(self.map.get_map()):
            #         return self.fish.vision, -1, True

            # update map and pseudo-3D rendering
            self.map.draw()
            self.fish.draw()
            self.fish2.draw()
            self.fish.cast_rays(self.map.map)
            reward += self.map.compute_reward(list(self.fish.position))

            text_to_print = self.reward_font.render(
                "Cumulative reward: " + str(round(self.cumulative_reward, 2)),
                True,
                (255, 255, 255),
                (85, 85, 85),
            )
            text_to_print_2 = self.mean_reward_font.render(
                "Mean reward: " + str(round(np.mean(last_rewards), 2)),
                True,
                (255, 255, 255),
                (85, 85, 85),
            )
            self.window.blit(text_to_print, (10, 10))
            self.window.blit(text_to_print_2, (250, 10))
            # update display
            pygame.display.flip()
            # set FPS
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return self.fish.vision / 255, -1, True

            if self.counter > 500:
                # reward += 10
                return self.fish.vision / 255, reward, True

        self.cumulative_reward += reward
        return self.fish.vision / 255, reward, False

    def run(self):
        done = False
        while not done:
            action = pygame.key.get_pressed()
            vision, reward, done = self.step(action)


if __name__ == "__main__":
    environement = Environment()
    environement.run()
