import random
from typing import Union

import numpy as np
import pygame

from RL_map_exploration.rl_environment.constants import MAP_SIZE, NBR_OF_FISHES
from RL_map_exploration.rl_environment.fish import Fish
from RL_map_exploration.rl_environment.map import TiledMap, TiledMapTopology


class Environment:
    def __init__(self) -> None:
        pygame.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((MAP_SIZE * 2, MAP_SIZE))

        pygame.display.set_caption("RL exploration game")
        self.reward_font = pygame.font.Font(None, 25)
        self.mean_reward_font = pygame.font.Font(None, 25)

        self.clock = pygame.time.Clock()

        map_topology = TiledMapTopology()
        for i in range(NBR_OF_FISHES):
            map_design, food_list = map_topology.get_topology()
            self.map = [
                TiledMap(self.window, map_design.copy(), food_list.copy())
                for i in range(NBR_OF_FISHES)
            ]
        self.fish = [
            Fish(self.map[i].get_map(), self.window, True if i == 0 else False)
            for i in range(NBR_OF_FISHES)
        ]
        self.counter = 0
        self.cumulative_reward = 0

    def reset(self):
        map_topology = TiledMapTopology()
        self.map = []
        for i in range(NBR_OF_FISHES):
            map_design, food_list = map_topology.get_topology()
            self.map = [
                TiledMap(self.window, map_design.copy(), food_list.copy())
                for i in range(NBR_OF_FISHES)
            ]
        self.fish = [
            Fish(self.map[i].get_map(), self.window, True if i == 0 else False)
            for i in range(NBR_OF_FISHES)
        ]
        self.clock.tick(0)
        self.counter = 0
        self.cumulative_reward = 0

        self.map[0].draw()
        for i, new_fish in enumerate(self.fish):
            new_fish.draw()
            new_fish.cast_rays(self.map[i].map, self.map[i].food_list)

        return [new_fish.vision / 255 for new_fish in self.fish]

    def step(
        self,
        action: Union[list[pygame.key.ScancodeWrapper], list[list]],  # type: ignore
        last_rewards: list = [0],
    ):
        reward = [0.0] * len(action)
        for _ in range(3):
            self.counter += 1
            pygame.draw.rect(
                self.window,
                (100, 100, 100),
                (MAP_SIZE, MAP_SIZE / 2, MAP_SIZE, MAP_SIZE),  # type: ignore
            )
            pygame.draw.rect(
                self.window,
                (150, 150, 150),
                (MAP_SIZE, -MAP_SIZE / 2, MAP_SIZE, MAP_SIZE),  # type: ignore
            )

            # handle user input
            if isinstance(action[0], list):
                left_index = 0
                right_index = 2
                up_index = 1
                left_forward_index = 3
                right_forward_index = 4
            else:
                left_index = pygame.K_LEFT
                right_index = pygame.K_RIGHT
                up_index = pygame.K_UP
                left_forward_index = pygame.K_LEFT
                right_forward_index = pygame.K_RIGHT

            for i, act in enumerate(action):
                if self.fish[i].done:
                    continue

                if act[left_index] or act[left_forward_index]:
                    self.fish[i].orientation += random.uniform(0.08, 0.12)
                if act[right_index] or act[right_forward_index]:
                    self.fish[i].orientation -= random.uniform(0.08, 0.12)
                if act[up_index] or act[left_forward_index] or act[right_forward_index]:
                    reward[i] += 0.0  # type: ignore
                    self.fish[i].position[0] += np.cos(self.fish[i].orientation) * 3
                    self.fish[i].position[1] += np.sin(self.fish[i].orientation) * 3
                    if self.fish[i].is_collision(self.map[i].get_map()):
                        # reward[i] -= 0.01
                        self.fish[i].position[0] += np.cos(self.fish[i].orientation) * 3
                        self.fish[i].position[1] += np.sin(self.fish[i].orientation) * 3
                        if self.fish[i].is_collision(self.map[i].get_map()):
                            self.fish[i].position[0] -= 2 * np.cos(self.fish[i].orientation) * 3
                            self.fish[i].position[1] -= 2 * np.sin(self.fish[i].orientation) * 3
                    # reward[i] = 0
                    # self.fish[i].done = True
                # else:
                #     reward[i] -= 0.001

            # update map and pseudo-3D rendering
            self.map[0].draw()
            for i, new_fish in enumerate(self.fish):
                new_fish.draw()
                new_fish.cast_rays(self.map[i].map, self.map[i].food_list)
                reward[i] += self.map[i].compute_reward(list(new_fish.position))

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
                print("Max step reached")
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
