import numpy as np
import pygame

from environment.constants import TILE_NUMBER, TILE_SIZE, MAP_SIZE


class TiledMap:
    def __init__(self, window) -> None:
        self.window = window
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

        for row in range(1, TILE_NUMBER - 1):
            for column in range(1, TILE_NUMBER - 1):
                if self.map[row, column] == 0:
                    self.map[row, column] = np.random.choice([0, 2], p=[0.8, 0.2])

    def get_map(self) -> np.ndarray:
        return self.map

    def draw(self) -> None:
        for row in range(TILE_NUMBER):
            for column in range(TILE_NUMBER):
                if self.map[row, column] == 1:
                    color = (85, 85, 85)
                elif self.map[row, column] == 2:
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 255)
                pygame.draw.rect(
                    self.window, color, (column * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                )

    def compute_reward(self, palyer_position: list) -> int:
        tile_position = np.array(
            [
                int((MAP_SIZE - palyer_position[1]) // TILE_SIZE),
                int(palyer_position[0] // TILE_SIZE),
            ]
        )
        if self.map[tile_position[0], tile_position[1]] == 2:
            self.map[tile_position[0], tile_position[1]] = 0
            return 0
        else:
            return 0
