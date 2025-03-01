import numpy as np
import pygame

from environment.constants import (
    FIELD_OF_VIEW,
    FISH_DIAMETER,
    MAP_SIZE,
    NUMBER_OF_RAYS,
    TILE_NUMBER,
    TILE_SIZE,
)
from environment.ray import Ray


class Fish:
    def __init__(self, map, window) -> None:
        self.window = window
        possible_tiles = []
        for row in range(1, TILE_NUMBER - 1):
            for column in range(1, TILE_NUMBER - 1):
                if (
                    map[row, column] == 0
                    and map[row - 1, column] == 0
                    and map[row, column - 1] == 0
                    and map[row + 1, column] == 0
                    and map[row, column + 1] == 0
                ):

                    possible_tiles.append([row, column])

        try:
            self.position = (
                np.array(possible_tiles[int(len(possible_tiles) / 2)], dtype=float) * TILE_SIZE
            )
            self.position += int(TILE_SIZE / 2)
            # change axis convention to classic one
            self.position = [self.position[1], MAP_SIZE - self.position[0]]
        except:
            self.position = [TILE_SIZE, TILE_SIZE]

        self.orientation = np.random.uniform(0, np.pi)

        self.vision = np.zeros((NUMBER_OF_RAYS, 3))

    def draw(self) -> None:
        # try:
        player_position_on_window = [int(self.position[0]), int(MAP_SIZE - self.position[1])]
        # except:
        #     self.postion = [TILE_SIZE, TILE_SIZE]
        #     player_position_on_window = [int(self.position[0]), int(MAP_SIZE - self.position[1])]
        pygame.draw.circle(self.window, (255, 0, 0), player_position_on_window, FISH_DIAMETER)

    def cast_rays(self, map) -> None:
        ray_angles = np.linspace(
            self.orientation - FIELD_OF_VIEW / 2,
            self.orientation + FIELD_OF_VIEW / 2,
            NUMBER_OF_RAYS,
        )

        for index, angle in enumerate(ray_angles):
            ray = Ray(self.position, angle, self.orientation, map, index, self.window)
            end_point, color = ray.cast_ray()
            self.vision[index, :] = np.array(color)

            # show field of view
            if index == 0 or index == NUMBER_OF_RAYS - 1:
                player_position_on_window = [
                    int(self.position[0]),
                    int(MAP_SIZE - self.position[1]),
                ]
                ray_end_point = [int(end_point[0]), int(MAP_SIZE - end_point[1])]
                pygame.draw.line(self.window, (0, 0, 255), player_position_on_window, ray_end_point)

    def is_collision(self, map):
        tile_position = (
            int((MAP_SIZE - self.position[1]) // TILE_SIZE),
            int(self.position[0] // TILE_SIZE),
        )
        player_position_on_tile = np.array(
            [(MAP_SIZE - self.position[1]) / TILE_SIZE, self.position[0] / TILE_SIZE], dtype=float
        )

        if map[tile_position] == 1:
            return True

        fish_bounding_box = [
            player_position_on_tile[1] - FISH_DIAMETER / TILE_SIZE * 0.9,
            player_position_on_tile[1] + FISH_DIAMETER / TILE_SIZE * 0.9,
            player_position_on_tile[0] + FISH_DIAMETER / TILE_SIZE * 0.9,
            player_position_on_tile[0] - FISH_DIAMETER / TILE_SIZE * 0.9,
        ]

        tile_check_deltas = [-1, 0, 1]
        for delta_x in tile_check_deltas:
            for delta_y in tile_check_deltas:
                tile_checked_position = (tile_position[0] - delta_x, tile_position[1] - delta_y)
                if map[tile_checked_position] == 1:
                    # bounding box in format [left, right, bottom, top]
                    tile_bounding_box = [
                        tile_checked_position[1],
                        tile_checked_position[1] + 1,
                        tile_checked_position[0] + 1,
                        tile_checked_position[0],
                    ]

                    x_overlaps = (tile_bounding_box[0] < fish_bounding_box[1]) and (
                        tile_bounding_box[1] > fish_bounding_box[0]
                    )
                    y_overlaps = (tile_bounding_box[3] < fish_bounding_box[2]) and (
                        tile_bounding_box[2] > fish_bounding_box[3]
                    )
                    collision = x_overlaps and y_overlaps

                    if collision:
                        return True
        return False
