import cv2
import numpy as np


def x_distance_between(self, other):
    distance = other.position[0] - self.position[0]
    return distance


def y_distance_between(self, other):
    distance = other.position[1] - self.position[1]
    return distance


def xy_distance_between(self, other):
    distance = other.position - self.position
    distance = np.linalg.norm(distance)
    return distance


def is_within_x_range(self, other, start, stop, absolute=False):
    distance = other.position[0] - self.position[0]
    if absolute:
        distance = abs(distance)
    condition = start < distance < stop
    return condition


def is_within_y_range(self, other, start, stop, absolute=False):
    distance = other.position[1] - self.position[1]
    if absolute:
        distance = abs(distance)
    condition = start < distance < stop
    return condition


def is_within_xy_magnitude_range(self, other, start, stop):
    distance = other.position - self.position
    distance = np.linalg.norm(distance)
    condition = start < distance < stop
    return condition


def is_intersect(self, other):
    object_1 = self.hull if self.hull is not None else self.position
    object_2 = other.hull if other.hull is not None else other.position

    canvas = np.zeros(self.canvas_size)
    # image1 = cv2.drawContours(canvas.copy(), np.expand_dims(object_1, 1), 0, 1)
    # image2 = cv2.drawContours(canvas.copy(), np.expand_dims(object_2, 1), 0, 1)
    image1 = cv2.fillPoly(canvas.copy(), [object_1], 1)
    image2 = cv2.fillPoly(canvas.copy(), [object_2], 1)

    intersection = np.logical_and(image1, image2).any()

    return intersection


def is_in_x_line_of(self, other, self_y_shorten=0):
    self_y_extremes = (np.min(self.hull, axis=0)[1], np.max(self.hull, axis=0)[1])
    self_y_extremes = (self_y_extremes[0] + self_y_shorten, self_y_extremes[1] - self_y_shorten)
    other_y_extremes = (np.min(other.hull, axis=0)[1], np.max(other.hull, axis=0)[1])

    # Assumes that "other" is smaller and is engulfed within "self - shortened"
    if other_y_extremes[0] > self_y_extremes[0] and other_y_extremes[1] < self_y_extremes[1]:
        return True
    else:
        return False


def is_right_of(self, other, hull=False):
    if hull:
        condition = np.max(other.hull, axis=0)[0] - np.min(self.hull, axis=0)[0]
    else:
        condition = other.position[0] - self.position[0]
    return condition < 0


def is_left_of(self, other, hull=False):
    if hull:
        condition = np.min(other.hull, axis=0)[0] - np.max(self.hull, axis=0)[0]
    else:
        condition = other.position[0] - self.position[0]
    return condition > 0


def is_top_of(self, other, hull=False):
    if hull:
        condition = np.min(other.hull, axis=0)[1] - np.max(self.hull, axis=0)[1]
    else:
        condition = other.position[1] - self.position[1]
    return condition > 0


def is_bottom_of(self, other, hull=False):
    if hull:
        condition = np.max(other.hull, axis=0)[1] - np.min(self.hull, axis=0)[1]
    else:
        condition = other.position[1] - self.position[1]
    return condition < 0
