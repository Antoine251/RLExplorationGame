import numpy as np
import pygame

from RL_map_exploration.rl_environment.constants import (
    FOOD_DIAMETER,
    MAP_SIZE,
    SCALE_FACTOR_3D,
    TILE_SIZE,
)


class Ray:
    def __init__(self, init_point, angle, player_angle, map, food_list, index, window) -> None:
        self.window = window
        self.index = index
        self.init_point = init_point
        self.end_point = init_point
        self.angle = angle
        self.player_angle = player_angle
        self.map = map
        self.food_list = food_list
        self.hit_food = False

    def find_wall(self) -> None:
        direction = np.array([np.cos(self.angle), np.sin(self.angle)], dtype=float)
        unit_step_size = np.array(
            [
                np.sqrt(1 + (direction[1] / direction[0]) ** 2),
                np.sqrt(1 + (direction[0] / direction[1]) ** 2),
            ]
        )
        tile_position = np.array(
            [
                int((MAP_SIZE - self.init_point[1]) // TILE_SIZE),
                int(self.init_point[0] // TILE_SIZE),
            ]
        )
        player_position_on_tile = np.array(
            [(MAP_SIZE - self.init_point[1]) / TILE_SIZE, self.init_point[0] / TILE_SIZE],
            dtype=float,
        )
        ray_length_1D = np.ones(2)
        depth = 0.0

        step = np.ones(2)
        if direction[0] < 0:
            step[1] = -1
            ray_length_1D[0] = (player_position_on_tile[1] - tile_position[1]) * unit_step_size[0]
        else:
            step[1] = 1
            ray_length_1D[0] = (tile_position[1] + 1 - player_position_on_tile[1]) * unit_step_size[
                0
            ]
        if direction[1] < 0:
            step[0] = 1
            ray_length_1D[1] = (tile_position[0] + 1 - player_position_on_tile[0]) * unit_step_size[
                1
            ]
        else:
            step[0] = -1
            ray_length_1D[1] = (player_position_on_tile[0] - tile_position[0]) * unit_step_size[1]

        tile_found = False
        if self.map[tile_position[0], tile_position[1]] == 2:
            ray_length_1D_save = ray_length_1D.copy()
            tile_position_save = tile_position.copy()
            if ray_length_1D[0] < ray_length_1D[1]:
                tile_position[1] += step[1]
                depth = ray_length_1D[0]
                ray_length_1D[0] += unit_step_size[0]
            else:
                tile_position[0] += step[0]
                depth = ray_length_1D[1]
                ray_length_1D[1] += unit_step_size[1]
            player_point = [self.init_point[0], MAP_SIZE - self.init_point[1]]
            distance = direction * depth
            end_point_on_tile = player_position_on_tile + np.array([-distance[1], distance[0]])
            next_tile_end_point = np.array(
                [end_point_on_tile[1] * TILE_SIZE, end_point_on_tile[0] * TILE_SIZE]
            )
            # pygame.draw.circle(self.window, (255, 255, 0), player_point, 3)
            # pygame.draw.circle(self.window, (255, 255, 0), next_tile_end_point, 3)
            tile_found = self.investigate_tile_for_food(np.array(player_point), next_tile_end_point)
            if tile_found is True:
                self.hit_food = True

            ray_length_1D = ray_length_1D_save
            tile_position = tile_position_save

        while tile_found is False and depth < MAP_SIZE * np.sqrt(2):
            if ray_length_1D[0] < ray_length_1D[1]:
                tile_position[1] += step[1]
                depth = ray_length_1D[0]
                ray_length_1D[0] += unit_step_size[0]
            else:
                tile_position[0] += step[0]
                depth = ray_length_1D[1]
                ray_length_1D[1] += unit_step_size[1]

            if tile_position[0] == 12:
                tile_position[0] = 11
            if tile_position[1] == 12:
                tile_position[1] = 11
            if self.map[tile_position[0], tile_position[1]] == 1:
                tile_found = True
                self.hit_food = False
            if self.map[tile_position[0], tile_position[1]] == 2:
                distance = direction * depth
                end_point_on_tile = player_position_on_tile + np.array([-distance[1], distance[0]])
                end_point = np.array(
                    [end_point_on_tile[1] * TILE_SIZE, end_point_on_tile[0] * TILE_SIZE]
                )

                ray_length_1D_save = ray_length_1D.copy()
                tile_position_save = tile_position.copy()

                if ray_length_1D[0] < ray_length_1D[1]:
                    tile_position[1] += step[1]
                    depth = ray_length_1D[0]
                    ray_length_1D[0] += unit_step_size[0]
                else:
                    tile_position[0] += step[0]
                    depth = ray_length_1D[1]
                    ray_length_1D[1] += unit_step_size[1]

                distance = direction * depth
                end_point_on_tile = player_position_on_tile + np.array([-distance[1], distance[0]])
                next_tile_end_point = np.array(
                    [end_point_on_tile[1] * TILE_SIZE, end_point_on_tile[0] * TILE_SIZE]
                )

                # pygame.draw.circle(self.window, (255, 255, 0), end_point, 3)
                # pygame.draw.circle(self.window, (255, 255, 0), next_tile_end_point, 3)
                tile_found = self.investigate_tile_for_food(end_point, next_tile_end_point)
                if tile_found is True:
                    self.hit_food = True

                ray_length_1D = ray_length_1D_save
                tile_position = tile_position_save

        if tile_found and self.hit_food is False:
            distance = direction * depth
            end_point_on_tile = player_position_on_tile + np.array([-distance[1], distance[0]])
            self.end_point = np.array(
                [end_point_on_tile[1] * TILE_SIZE, MAP_SIZE - end_point_on_tile[0] * TILE_SIZE]
            )

    def investigate_tile_for_food(
        self, start_point: np.ndarray, end_point: np.ndarray, step_size: float = 2
    ) -> bool:
        """Check if there is food on the tile. If yes, find the collision point."""
        direction_forward = (end_point - start_point) / np.linalg.norm(end_point - start_point)

        # Initialize moving points
        forward_point = start_point.copy()
        backward_point = end_point.copy()

        mid_point = (forward_point + backward_point) / 2
        food = None
        for food_center in self.food_list:
            food_position = np.array(food_center)
            food_tile = food_position // TILE_SIZE
            if np.all(mid_point // TILE_SIZE == food_tile):
                food = food_center

        if food is None:
            return False

        while np.linalg.norm(forward_point - backward_point) > step_size:
            # Move forward point
            forward_point += direction_forward * step_size
            if np.linalg.norm(forward_point - food) < FOOD_DIAMETER / 2:
                end_point_on_tile = forward_point
                self.end_point = np.array([end_point_on_tile[0], MAP_SIZE - end_point_on_tile[1]])
                return True

        return False

    def cast_ray(self, draw: bool) -> tuple[np.ndarray, tuple]:  # [2,], (3,)
        self.find_wall()
        color = self.show_3D_map(draw)

        return self.end_point, color

    def show_3D_map(self, draw: bool) -> tuple:
        depth = np.sqrt(
            (self.init_point[0] - self.end_point[0]) ** 2
            + (self.init_point[1] - self.end_point[1]) ** 2
        )
        depth_color = 255 / (1 + depth**2 * 0.0001)

        if self.hit_food is True:
            color = (depth_color, 0, 0)
        else:
            color = (0, depth_color, depth_color)

        # remove fish eye effect
        if draw:
            depth *= np.cos(self.player_angle - self.angle)

            wall_height = 21000 / (depth + 0.0001)
            pygame.draw.rect(
                self.window,
                color,
                (
                    (2 * MAP_SIZE) - (self.index + 1) * SCALE_FACTOR_3D,
                    (MAP_SIZE / 2) - wall_height / 2,
                    SCALE_FACTOR_3D + 1,
                    wall_height,
                ),  # type: ignore
            )

        return color
