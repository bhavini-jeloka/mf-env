import numpy as np
import random


def opacity(color: tuple, alpha=0.5):
    '''
    Function to alter opacity based on scaling factor alpha.
    '''
    return (255 - alpha * (255 - color[0]), 255 - alpha * (255 - color[1]), 255 - alpha * (255 - color[2]))


def place_circle(n, r, x0, y0, L):
    '''
    Function to place circles within the square with reduced overlap
    :param n: number of circles
    :param r: radius of circles
    :param x0: x coordinate of bottom-left corner of the square
    :param y0: y coordinate of bottom-left corner of the square
    :param L: side length of the square
    :return: list of center positions of the circles
    '''
    # Function to calculate the distance between two points
    def distance(x1, y1, x2, y2):
        """Compute the Euclidean distance between two points."""
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Function to check if two circles overlap
    def overlap(x1, y1, x2, y2, r):
        """Return whether two circles overlap."""
        return distance(x1, y1, x2, y2) < 2 * r

    # Function to adjust circle positions to reduce overlap
    def adjust_positions(circles, r, L, x0, y0):
        """Move circles as needed to avoid overlaps within the drawing area."""
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                x1, y1 = circles[i]
                x2, y2 = circles[j]
                if overlap(x1, y1, x2, y2, r):
                    # Move circle j away from circle i
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = distance(x1, y1, x2, y2)
                    if dist != 0:
                        move_dist = (2 * r - dist) / 2
                        dx *= move_dist / dist
                        dy *= move_dist / dist
                        circles[j] = (x2 + dx, y2 + dy)
                        # Ensure circles remain within the square
                        circles[j] = (max(x0 + r, min(x0 + L - r, circles[j][0])),
                                      max(y0 + r, min(y0 + L - r, circles[j][1])))
        return circles

    circles = []
    for _ in range(n):
        n_attempts = 0
        while True:
            x = random.uniform(x0 + r, x0 + L - r)
            y = random.uniform(y0 + r, y0 + L - r)
            new_circle = (x, y)

            # Check if this circle overlaps too much with any existing circle
            if all(not overlap(x, y, cx, cy, r) for cx, cy in circles):
                circles.append(new_circle)
                break
            n_attempts += 1

            if n_attempts >= 10:
                circles.append(new_circle)
                break

    # Adjust positions to minimize overlap
    for _ in range(20):  # Iterate to reduce overlap
        circles = adjust_positions(circles, r, L, x0, y0)

    return circles
