import glob
import time

import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from random import shuffle
import matplotlib.image as mpimg
from torch.utils.data import Dataset
import itertools
import torch
from inspect import currentframe, getframeinfo
from PIL import Image


class shape(list):

    def __init__(self):
        self.append(np.array((  # triangle
            (0.5, 0.0),
            (-0.25, 0.5 * np.sin(2 * np.pi / 3)),
            (-0.25, 0.5 * np.sin(4 * np.pi / 3)),
        )))

        self.append(np.array((  # square
            (0.3535533905932738, 0.3535533905932738),
            (-0.35355339059327373, 0.3535533905932738),
            (-0.35355339059327384, -0.35355339059327373),
            (0.3535533905932737, -0.35355339059327384),
        )))

        self.append(np.array((  # hexagon
            (0.15450849718747373, 0.47552825814757677),
            (-0.40450849718747367, 0.2938926261462366),
            (-0.4045084971874737, -0.2938926261462365),
            (0.15450849718747361, -0.4755282581475768),
            (0.5, -1.2246467991473532e-16),
        )))

        self.append(np.array((  # star
            (0.15450849718747373, 0.47552825814757677),
            (-0.06180339887498947, 0.19021130325903074),
            (-0.40450849718747367, 0.2938926261462366),
            (-0.2, 2.4492935982947065e-17),
            (-0.4045084971874737, -0.2938926261462365),
            (-0.061803398874989514, -0.1902113032590307),
            (0.15450849718747361, -0.4755282581475768),
            (0.16180339887498948, -0.11755705045849468),
            (0.5, -1.2246467991473532e-16),
            (0.1618033988749895, 0.11755705045849459),
        )))

        self.append(0.5)  # circle


class position(list):
    def __init__(self, grid_size):

        for i in range(3):
            for j in range(3):
                x = int(np.floor((0.5 + i) * grid_size / 3))
                y = int(np.floor((0.5 + j) * grid_size / 3))
                self.append(np.array((x, y)))


class size:
    def __init__(self, grid_size, isrule=False, **kwargs):
        self.grid = grid_size
        self.min = 5 / 100 * self.grid
        self.max = self.grid * 0.9 / 3
        self.alpha = kwargs.get('alpha', None)

        if isrule:
            if self.alpha == None:
                print('Size class: missing alpha')
                exit()
            elif np.abs(self.alpha) > 1.:
                print('Size class: abs(alpha) larger than 1')
                exit()
            self.df = self.alpha * (self.max - self.min)
            if self.alpha > 0:
                self.max -= self.df
            else:
                self.min -= self.df

    def sample(self):
        return np.random.uniform(self.min, self.max)


class color:
    def __init__(self, isrule=False, **kwargs):

        self.min = 0.15
        self.max = 1
        self.alpha = kwargs.get('alpha', None)

        if isrule:
            if self.alpha == None:
                print('Color class: missing alpha')
                exit()
            elif np.abs(self.alpha) > 1.:
                print('Color class: abs(alpha) larger than 1')
                exit()
            self.df = self.alpha * (self.max - self.min)
            if self.alpha > 0:
                self.max -= self.df
            else:
                self.min -= self.df

    def sample(self):
        return np.random.uniform(self.min, self.max)


class number:
    def __init__(self, isrule=False, **kwargs):

        self.min = 1
        self.max = 9
        self.alpha = kwargs.get('alpha', None)

        if isrule:
            if self.alpha == None:
                print('Number class: missing alpha')
                exit()
            elif np.abs(self.alpha) > 1.:
                print('Number class: abs(alpha) larger than 1')
                exit()

            self.df = np.round(self.alpha * (self.max - self.min)).astype(int)
            if self.alpha > 0:
                self.max -= self.df
            else:
                self.min -= self.df

    def sample(self):
        return np.random.randint(self.min, self.max + 1)


def initialize_feature_generators(HP):
    rules, grid_size, alpha = HP['rules'], HP['grid_size'], HP['alpha']

    shapes_g = shape()

    positions_g = position(grid_size)

    if rules['size'] == 2:
        sizes_g = size(grid_size, isrule=True, alpha=alpha)
    elif rules['size'] == 0 or rules['size'] == 1:
        sizes_g = size(grid_size)
    elif isinstance(rules["size"], str):
        sizes_g = (HP['grid_size'] * 0.9 / 3 * float(rules['size']),) * 2

    if rules['color'] == 2:
        colors_g = color(isrule=True, alpha=alpha)
    elif rules['color'] == 0 or rules['color'] == 1:
        colors_g = color()
    elif isinstance(rules["color"], str):
        colors_g = (float(rules['color']),) * 2

    if rules['number'] == 2:
        numbers_g = number(isrule=True, alpha=alpha)
    elif rules['number'] == 0 or rules['number'] == 1:
        numbers_g = number()
    elif isinstance(rules["number"], str):
        numbers_g = (int(rules['number']),) * 2

    return shapes_g, positions_g, colors_g, numbers_g, sizes_g



def sample_features(HP, shapes_g, positions_g, colors_g, numbers_g, sizes_g):
    rules, grid_size, alpha = HP['rules'], HP['grid_size'], HP['alpha']
    if isinstance(rules["shape"], str):
        shapes = [shapes_g[int(rules["shape"])]]
    else:
        shapes = shapes_g.copy()
        shuffle(shapes)

    positions = positions_g.copy()
    shuffle(positions)

    if rules['size'] == 2:
        size1 = sizes_g.sample()
        sizes = (size1, size1 + sizes_g.df)
    elif rules['size'] == 1:
        sizes = (sizes_g.sample(), sizes_g.sample())
    elif rules['size'] == 0:
        sizes = (sizes_g.sample(),) * 2
    elif isinstance(rules["size"], str):
        sizes = sizes_g

    if rules['color'] == 2:
        color1 = colors_g.sample()
        colors = (color1, color1 + colors_g.df)
    elif rules['color'] == 1:
        colors = (colors_g.sample(), colors_g.sample())
    elif rules['color'] == 0:
        colors = (colors_g.sample(),) * 2
    elif isinstance(rules["color"], str):
        colors = colors_g

    if rules['number'] == 2:
        number1 = numbers_g.sample()
        numbers = (number1, number1 + numbers_g.df)
    elif rules['number'] == 1:
        numbers = (numbers_g.sample(), numbers_g.sample())
    elif rules['number'] == 0:
        numbers = (numbers_g.sample(),) * 2
    elif isinstance(rules["number"], str):
        numbers = numbers_g

    return shapes, positions, colors, numbers, sizes


def one_example(HP, shapes_g, positions_g, colors_g, numbers_g, sizes_g):
    shapes, positions, colors, numbers, sizes = sample_features(HP, shapes_g, positions_g, colors_g, numbers_g, sizes_g)

    tiles = []
    for i in range(2):
        tile = np.zeros((HP['grid_size'], HP['grid_size']))
        for j in range(numbers[i]):
            if type(shapes[0]) is not float:
                rr, cc = draw.polygon(positions[j][0] + sizes[i] * shapes[0][:, 0],
                                      positions[j][1] + sizes[i] * shapes[0][:, 1],
                                      [HP['grid_size'], HP['grid_size']])
                tile[rr, cc] = colors[i]
            else:
                rr, cc = draw.disk((positions[j][0], positions[j][1]),
                                   sizes[i] * shapes[0], shape=[HP['grid_size'], HP['grid_size']])
                tile[rr, cc] = colors[i]

        if HP['channels'] > 1:
            tile = np.tile(np.expand_dims(tile, axis=0), (HP['channels'], 1, 1))

        tiles.append(tile)

        if HP['rules']["shape"] == 1:
            shuffle(shapes)

        if HP['rules']["position"] == 1:
            shuffle(positions)

    tiles = np.array(tiles)

    if HP['plot_examples']:
        fig, ax = plt.subplots(1, 2)
        cmap = 'gray'

        if HP['channels'] > 1:
            tiles = [t[0] for t in tiles]

        for j in range(int(HP['plot_evaluation']) + 1):
            for ind, t in enumerate(tiles):
                ax[ind].imshow(1 - tiles[ind], origin='lower', interpolation='none', vmin=0, vmax=1, cmap=cmap)
                # ax[ind].set_xlabel('Image ' + str(ind + 1), fontsize=fontsize, labelpad=labelpad)
                ax[ind].set_yticklabels([])
                ax[ind].set_xticklabels([])
                ax[ind].set_yticks([])
                ax[ind].set_xticks([])
                [k.set_linewidth(1) for k in ax[ind].spines.values()]

            plt.subplots_adjust(wspace=-0.05, hspace=0.5)
            plt.tight_layout()
            fname = f"alpha={HP['alpha']}_rule={tuple(HP['rules'].values())}"
            fname += '_' + str(len(glob.glob(fname + '*')))
            plt.savefig(fname + ".png", dpi=400, transparent=True)
            tiles = np.flip(tiles, axis=0)

    return tiles


def create_batch(HP = None):
    if HP is None:
        HP = {'grid_size': 224, 'channels': 3, 'plot_examples': True, 'plot_evaluation': False,
              'rules': {"color": 0,
                        "position": 0,
                        "size": 2,
                        "shape": 0,
                        "number": 0},
              'alpha': 0.5,
              'batch': 5}

    shapes_g, positions_g, colors_g, numbers_g, sizes_g = initialize_feature_generators(HP)

    images = []
    for i in range(HP['batch']):
        images.append(one_example(HP, shapes_g, positions_g, colors_g, numbers_g, sizes_g))
    images = torch.tensor(np.array(images)).float()
    return images


def plot_batch(fname_base):
    fname_base = fname_base[:fname_base.find('.png') - 2]

    fnames = glob.glob(fname_base + '*.png')
    fnames.sort()
    fig = plt.figure()
    angle_degrees = 65
    start_x = 0.2  # Adjust as needed
    start_y = 0.4  # Adjust as needed
    angle_radians = np.deg2rad(angle_degrees)
    overlap = 0.08
    offset_x = overlap * np.cos(angle_radians)
    offset_y = overlap * np.sin(angle_radians)

    for i, fname in enumerate(fnames):
        image = plt.imread(fname)
        x = start_x + i * offset_x
        y = start_y - i * offset_y
        ax = fig.add_axes([x, y, 0.5, 0.5])
        ax.imshow(image, aspect='auto')
        ax.axis('off')
    plt.axis('off')
    plt.savefig(fname_base + '_batch.png', dpi=400, transparent=True)



if __name__ == "__main__":
    HP = {'grid_size': 224, 'channels': 3, 'plot_examples': True, 'plot_evaluation': False,
          'rules': {"color": 0,
                    "position": 0,
                    "size": 2,
                    "shape": 0,
                    "number": 0},
          'alpha': 0.5,
          'batch': 5}

    create_batch(HP)

    plot_batch(f"alpha={0.5}_rule=(0, 0, 2, 0, 0)_0.png")
