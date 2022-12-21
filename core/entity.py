import numpy as np


class Velocity:
    def __init__(self, direction=None, magnitude=None):
        if direction is not None and magnitude is not None:
            self.direction = direction / np.linalg.norm(direction)
            self.magnitude = magnitude
        else:
            self.direction = np.zeros((1, 2))
            self.magnitude = 0.0

    def update_velocity(self, direction, time_since_last_update):
        if np.all(direction == 0):
            self.direction = direction
        else:
            self.direction = direction / np.linalg.norm(direction)
        if time_since_last_update > 0:
            self.magnitude = np.linalg.norm(direction) / time_since_last_update
        else:
            self.magnitude = 0


class Entity:
    def __init__(self, canvas_size, current_time, position=None, hull=None, direction=None, speed=None, **kwargs):
        self.canvas_size = canvas_size
        self.last_updated_time = current_time
        self.position = position
        self.hull = hull
        if hull is not None and position is None:
            self.position = np.mean(hull, axis=0)

        self.velocity = Velocity(direction, speed)

        # self.attended_to = False
        self.static = False  # Planes that are static (such as the ground plane)
        self.is_protagonist = False  # Main entity of the game

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.in_scope = True  # Still in game? Or destroyed/left frame.

    def update_position(self, new_position=None, new_hull=None, current_time=None):
        if new_hull is not None and new_position is None:
            new_position = np.mean(new_hull, axis=0)
            self.hull = new_hull
        difference = new_position - self.position
        if current_time is not None:
            # TODO: Velocity system possibly broken
            self.velocity.update_velocity(difference, current_time - self.last_updated_time)
            self.last_updated_time = current_time

        self.position = new_position
