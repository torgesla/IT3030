from configparser import ConfigParser
import numpy as np
from matplotlib import pyplot as plt
import random
import math
import itertools
import raster_geometry as rg

config = ConfigParser()
config.read('config.ini')
datagen = config['datagen']
BACKGROUND_SIZE = int(datagen['background_size'])  # 10 < N < 50
OFFSET = int(datagen['offset'])  # Offset from img border
LINE_WIDTH = int(datagen['line_width'])
IMAGE_SIZE = int(datagen['image_size'])
NOISE_LEVEL = float(datagen['noise_level'])  # Fraction of pixels 0-1


class DataGenerator():
    id_class = {0: 'Cross',
                1: 'Rectangle',
                2: 'Circle',
                3: 'Triangle'}

    def display_N_pictures(self, N):
        samples = random.sample(self.cases_labels, N)
        for vec, onehot in samples:
            img = vec.reshape(BACKGROUND_SIZE, BACKGROUND_SIZE)
            class_id = DataGenerator.onehot_to_index(onehot)
            class_name = DataGenerator.id_class[class_id]
            plt.imshow(img)
            plt.title(class_name)
            plt.show()

    @staticmethod
    def index_to_onehot(index):
        onehot = [0 for _ in range(len(DataGenerator.id_class))]
        onehot[index] = 1
        return onehot

    @staticmethod
    def onehot_to_index(on_hot):
        return list(on_hot).index(1.0)

    def make_cases(self, number_of_cases, train_val_test_split=(0.70, 0.15, 0.15), flatten=True):
        """
        Returns generated data for training. validation and testing

        :param tuple train_val_test_split: fraction of samples used for train, val and test phase
        :param bool flatten: decide if samples should be returned in flattened format or not
        returns (trainX,trainY, valX, valY, testX, testY)
        """
        if(sum(train_val_test_split) != 1.0):
            raise Exception('Split fractions does not add up to 1.0')
        case_base = []
        labels = []
        number_of_classes = len(Image.__subclasses__())
        per_class = number_of_cases // number_of_classes
        extra_for_last_class = number_of_cases % number_of_classes
        for _ in range(per_class):
            case_base.append(Cross().image)
            labels.append(self.index_to_onehot(0))

            case_base.append(Rectangle().image)
            labels.append(self.index_to_onehot(1))

            case_base.append(Circle().image)
            labels.append(self.index_to_onehot(2))

            case_base.append(Triangle().image)
            labels.append(self.index_to_onehot(3))

        for _ in range(extra_for_last_class):
            case_base.append(Triangle().image)
            labels.append(self.index_to_onehot(3))

        if(flatten):
            case_base = [x.flatten() for x in case_base]
        c = list(zip(case_base, labels))
        random.shuffle(c)
        case_base, labels = zip(*c)
        case_base, labels = np.stack(case_base, axis=0), np.stack(labels, axis=0)
        self.cases_labels = list(zip(case_base, labels))
        train_index = int(number_of_cases * train_val_test_split[0])
        val_index = train_index + int(number_of_cases * train_val_test_split[1])

        return (case_base[:train_index], labels[:train_index],
                case_base[train_index:val_index], labels[train_index:val_index],
                case_base[val_index:], labels[val_index:])


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
            if(col_gap > 3 and row_gap > 3):
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
    pass
