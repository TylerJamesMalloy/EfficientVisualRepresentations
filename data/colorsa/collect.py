import math

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd 
import imageio

import cv2
import imageio

from os import listdir
from os.path import isfile, join

colors = ['green.png', 'orange.png', 'purple.png']
all_data = []

for color in colors:
    im = imageio.imread('./raw/' + color)
    im = cv2.resize(im, (64, 64))
    data = np.asarray(im)
    all_data.append(data)

np.save("stimuli.npy", all_data)