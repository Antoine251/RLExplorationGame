import numpy as np
import pygame

from RL_map_exploration.rl_environment.constants import (
    FISH_DIAMETER,
    FOOD_DIAMETER,
    MAP_SIZE,
    TILE_NUMBER,
    TILE_SIZE,
)


class TiledMapTopology:
    def __init__(self) -> None:
        self.map = np.zeros((TILE_NUMBER, TILE_NUMBER))
        self.map[:, 0] = 1
        self.map[:, TILE_NUMBER - 1] = 1
        self.map[0, :] = 1
        self.map[TILE_NUMBER - 1, :] = 1
        for row in range(1, TILE_NUMBER - 1):
            for column in range(1, TILE_NUMBER - 1):

                # Init obstacles on edges
                if row == 1 or column == 1:
                    self.map[row, column] = np.random.choice([0, 1], p=[0.85, 0.15])
                    continue

                # Implement tiling rules
                left_tile = self.map[row - 1, column]
                above_tile = self.map[row, column - 1]

                if left_tile == 0 and above_tile == 0:
                    self.map[row, column] = np.random.choice([0, 1], p=[0.8, 0.2])
                if (left_tile == 1 and above_tile == 0) or (left_tile == 0 and above_tile == 1):
                    self.map[row, column] = np.random.choice([0, 1], p=[0.65, 0.35])
                if left_tile == 1 and above_tile == 1:
                    self.map[row, column] = np.random.choice([0, 1], p=[0.4, 0.6])

        # map clean: remove isolated walls and voids
        for row in range(1, TILE_NUMBER - 1):
            for column in range(1, TILE_NUMBER - 1):
                if (
                    self.map[row, column] == 1
                    and self.map[row - 1, column] == 0
                    and self.map[row, column - 1] == 0
                    and self.map[row + 1, column] == 0
                    and self.map[row, column + 1] == 0
                ):

                    self.map[row, column] = 0

                if (
                    self.map[row, column] == 0
                    and self.map[row - 1, column] == 1
                    and self.map[row, column - 1] == 1
                    and self.map[row + 1, column] == 1
                    and self.map[row, column + 1] == 1
                ):

                    self.map[row, column] = 1

        self.food_list = []
        for row in range(1, TILE_NUMBER - 1):
            for column in range(1, TILE_NUMBER - 1):
                if self.map[row, column] == 0:
                    self.map[row, column] = np.random.choice([0, 2], p=[0.9, 0.1])
                    if self.map[row, column] == 2:
                        x_position = (
                            np.random.rand() * (TILE_SIZE - FOOD_DIAMETER)
                            + column * TILE_SIZE
                            + FOOD_DIAMETER / 2
                        )
                        y_position = (
                            np.random.rand() * (TILE_SIZE - FOOD_DIAMETER)
                            + row * TILE_SIZE
                            + FOOD_DIAMETER / 2
                        )
                        self.food_list.append((x_position, y_position))

    def get_topology(self) -> tuple[np.ndarray, list]:
        return self.map, self.food_list


class TiledMap:
    def __init__(self, window, map_topology: np.ndarray, food_list: list) -> None:
        self.window = window
        self.map = map_topology
        self.initial_map = map_topology.copy()
        self.food_list = food_list

    def get_map(self) -> np.ndarray:
        return self.map

    def draw(self) -> None:
        for row in range(TILE_NUMBER):
            for column in range(TILE_NUMBER):
                if self.map[row, column] == 1:
                    color = (85, 85, 85)
                elif self.map[row, column] == 2:
                    color = (255, 255, 255)
                else:
                    color = (255, 255, 255)
                pygame.draw.rect(
                    self.window, color, (column * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                )

        # for row in range(TILE_NUMBER):
        #     for column in range(TILE_NUMBER):
        #         if self.initial_map[row, column] == 2:
        #             color = (255, 0, 0)
        #             pygame.draw.rect(
        #                 self.window,
        #                 color,
        #                 (column * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
        #                 width=1,
        #             )

        for food in self.food_list:
            pygame.draw.circle(self.window, (255, 100, 100), food, FOOD_DIAMETER / 2, 0)

    def compute_reward(self, player_position: list) -> int:
        tile_position = np.array(
            [
                int((MAP_SIZE - player_position[1]) // TILE_SIZE),
                int(player_position[0] // TILE_SIZE),
            ]
        )

        player_position = [player_position[0], MAP_SIZE - player_position[1]]

        for food_center in self.food_list:
            if (
                np.linalg.norm(np.array(player_position) - food_center)
                < (FOOD_DIAMETER / 2 + FISH_DIAMETER / 2) * 1.1
            ):
                self.map[tile_position[0], tile_position[1]] = 0
                self.food_list.remove(food_center)
                return 5

        return 0
