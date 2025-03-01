from typing import Union

import numpy as np
import pygame

from RL_map_exploration.rl_environment.constants import MAP_SIZE, NBR_OF_FISHES
from RL_map_exploration.rl_environment.fish import Fish
from RL_map_exploration.rl_environment.map import TiledMap


class Environment:
    def __init__(self) -> None:
        pygame.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((MAP_SIZE * 2, MAP_SIZE))

        pygame.display.set_caption("RL exploration game")
        self.reward_font = pygame.font.Font(None, 25)
        self.mean_reward_font = pygame.font.Font(None, 25)

        self.clock = pygame.time.Clock()

        self.map = TiledMap(self.window)
        self.fish = [Fish(self.map.get_map(), self.window) for _ in range(NBR_OF_FISHES)]
        self.counter = 0
        self.cumulative_reward = 0

    def reset(self):
        self.map = TiledMap(self.window)
        self.fish = [Fish(self.map.get_map(), self.window) for _ in range(NBR_OF_FISHES)]
        self.clock.tick(0)
        self.counter = 0
        self.cumulative_reward = 0

        self.map.draw()
        for i, new_fish in enumerate(self.fish):
            new_fish.draw(fainted=True if i > 0 else False)
            new_fish.cast_rays(self.map.map)

        return [new_fish.vision / 255 for new_fish in self.fish]

    def step(
        self,
        action: Union[list[pygame.key.ScancodeWrapper], list[list]],  # type:ignore
        last_rewards: list = [0],
    ):
        reward = [0] * len(action)
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
            if isinstance(action[0], list):
                left_index = 0
                right_index = 2
                up_index = 1
            else:
                left_index = pygame.K_LEFT
                right_index = pygame.K_RIGHT
                up_index = pygame.K_UP

            for i, act in enumerate(action):
                if act[left_index]:
                    self.fish[i].orientation += 0.1
                if act[right_index]:
                    self.fish[i].orientation -= 0.1
                if act[up_index]:
                    reward[i] += 0.2  # type: ignore
                    self.fish[i].position[0] += np.cos(self.fish[i].orientation) * 3
                    self.fish[i].position[1] += np.sin(self.fish[i].orientation) * 3
                    if self.fish[i].is_collision(self.map.get_map()):
                        reward[i] = -10
                        self.fish[i].done = True

            # if action[pygame.K_DOWN]:
            #     self.fish.position[0] -= np.cos(self.fish.orientation)*3
            #     self.fish.position[1] -= np.sin(self.fish.orientation)*3
            #     if self.fish.is_collision(self.map.get_map()):
            #         return self.fish.vision, -1, True

            # update map and pseudo-3D rendering
            self.map.draw()
            for i, new_fish in enumerate(self.fish):
                new_fish.draw(fainted=True if i > 0 else False)
                new_fish.cast_rays(self.map.map)
                reward[i] += self.map.compute_reward(list(new_fish.position))

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
                    for new_fish in self.fish:
                        new_fish.done = True
                    return [
                        (new_fish.vision / 255, -1, new_fish.done)
                        for i, new_fish in enumerate(self.fish)
                    ]

            if self.counter > 500:
                # reward += 10
                for new_fish in self.fish:
                    new_fish.done = True
                return [
                    (new_fish.vision / 255, reward[i], new_fish.done)
                    for i, new_fish in enumerate(self.fish)
                ]

        self.cumulative_reward += reward[0]
        return [
            (new_fish.vision / 255, reward[i], new_fish.done)
            for i, new_fish in enumerate(self.fish)
        ]

    def run(self):
        done = False
        while not done:
            action = [pygame.key.get_pressed() for _ in range(NBR_OF_FISHES)]
            _ = self.step(action)


if __name__ == "__main__":
    environement = Environment()
    environement.run()
