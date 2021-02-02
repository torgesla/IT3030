import numpy as np
from matplotlib import pyplot as plt
import random
import math
import itertools
import raster_geometry as rg
NUMBER_OF_CASES = 100
BACKGROUND_SIZE = 50  # 10 < N < 50
OFFSET = 1  # Offset from img border
LINE_WIDTH = 1
IMAGE_SIZE = 20
NOISE_LEVEL = 0.01  # Fraction of pixels 0-1


class DataGenerator():
    @staticmethod
    def make_cases():
        case_base = []
        # Change next two lines if number of classes changes
        per_class = NUMBER_OF_CASES//4
        extra_for_last_class = NUMBER_OF_CASES % 4
        for _ in range(per_class):
            case_base.append(Cross().image.flatten())
            case_base.append(Rectangle().image.flatten())
            case_base.append(Circle().image.flatten())
            case_base.append(Triangle().image.flatten())
        for _ in range(extra_for_last_class):
            case_base.append(Triangle().flatten())
        random.shuffle(case_base)
        return case_base


class Image():
    def __init__(self):
        self.image = np.zeros((BACKGROUND_SIZE, BACKGROUND_SIZE))

    def visualize(self):
        plt.imshow(self.image)
        plt.show()

    def add_noise(self):
        possible_indices = list(itertools.product(range(BACKGROUND_SIZE), range(BACKGROUND_SIZE)))
        total_number_of_cells = BACKGROUND_SIZE * BACKGROUND_SIZE
        noise_indices = random.sample(possible_indices, int(NOISE_LEVEL*total_number_of_cells))
        for r, c in noise_indices:
            self.image[r, c] = float(not bool(self.image[r, c]))


class Cross(Image):
    def __init__(self):
        super().__init__()
        self.generate_cross()
        super().add_noise()

    def generate_cross(self):
        rand_col = random.randint(OFFSET, BACKGROUND_SIZE - (OFFSET + LINE_WIDTH))
        rand_row = random.randint(OFFSET, BACKGROUND_SIZE - (OFFSET + LINE_WIDTH))
        self.image[rand_row:rand_row+LINE_WIDTH] = 1.0
        self.image[:, rand_col:rand_col+LINE_WIDTH] = 1.0


class Rectangle(Image):
    def __init__(self):
        super().__init__()
        self.generate_rectangle()
        super().add_noise()

    def generate_rectangle(self):
        while(True):
            rand_col_1 = random.randint(OFFSET, BACKGROUND_SIZE - (OFFSET + LINE_WIDTH))
            rand_col_2 = random.randint(OFFSET, BACKGROUND_SIZE - (OFFSET + LINE_WIDTH))

            rand_row_1 = random.randint(OFFSET, BACKGROUND_SIZE - (OFFSET + LINE_WIDTH))
            rand_row_2 = random.randint(OFFSET, BACKGROUND_SIZE - (OFFSET + LINE_WIDTH))
            col_gap = abs(rand_col_1 - rand_col_2)
            row_gap = abs(rand_row_1 - rand_row_2)
            if(col_gap > 2 and row_gap > 2):
                break
        min_row, max_row = min(rand_row_1, rand_row_2), max(rand_row_1, rand_row_2)
        min_col, max_col = min(rand_col_1, rand_col_2), max(rand_col_1, rand_col_2)
        self.image[min_row:max_row, min_col:max_col] = 1.0
        self.image[min_row+LINE_WIDTH:max_row-LINE_WIDTH, min_col+LINE_WIDTH:max_col-LINE_WIDTH] = 0.0  # Remove fill of rectangle, leaving the outer frame


class Circle(Image):
    def __init__(self, radius=IMAGE_SIZE//2):
        super().__init__()
        self.generate_circle(radius)
        super().add_noise()

    def generate_circle(self, radius):
        radius = IMAGE_SIZE//2
        origo_row = random.randint(OFFSET + radius, BACKGROUND_SIZE-(radius + OFFSET))
        origo_col = random.randint(OFFSET + radius, BACKGROUND_SIZE-(radius + OFFSET))
        for i, j in itertools.product(range(BACKGROUND_SIZE), range(BACKGROUND_SIZE)):
            if(int(math.hypot(origo_row - i, origo_col - j)) in range(radius-LINE_WIDTH//2, radius+LINE_WIDTH//2 + 1)):
                self.image[i, j] = 1.0


class Triangle(Image):
    def __init__(self):
        super().__init__()
        self.generate_triangle()
        super().add_noise()

    def generate_triangle(self):
        # coordinates for down left corner
        rand_col_dl = random.randint(OFFSET, BACKGROUND_SIZE - (IMAGE_SIZE + OFFSET))
        rand_row_dl = random.randint(OFFSET + IMAGE_SIZE, BACKGROUND_SIZE - OFFSET)
        dl = (rand_row_dl, rand_col_dl)
        # coordinates, down right
        row_dr = rand_row_dl
        col_dr = rand_col_dl + IMAGE_SIZE
        dr = (row_dr, col_dr)
        # coordinates, top point
        rand_col_top = random.randint(OFFSET, BACKGROUND_SIZE - (OFFSET))
        row_top = rand_row_dl - IMAGE_SIZE
        top = (row_top, rand_col_top)
        """ self.image[rand_row_dl, rand_col_dl:col_dr] = 1.0 """
        # vectors
        coords = set(rg.bresenham_lines((dl, dr, top), closed=True))
        for r, w in coords:
            self.image[r, w] = 1.0


if __name__ == "__main__":
    x = DataGenerator.make_cases()
    for fig in x[:2]:
        print(list(fig))
