# Plot an example of the possible of a shift in a B-VAE model after learning utility 
import argparse
from cmath import nan
import enum
import logging
from re import L
import sys
import os
import copy 
from configparser import ConfigParser

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from disvae import init_specific_model, Trainer, Evaluator
from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.losses import UTIL_LOSSES, LOSSES, RECON_DIST, get_loss_f
from disvae.models.vae import MODELS, UTILTIIES 
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, update_namespace_, FormatterNoDuplicate)
from utils.visualize import GifTraversalsTraining

from torch.utils.data import Dataset, DataLoader

import torch 
from torch import optim
import pandas as pd 
import numpy as np 
from scipy import io 
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import seaborn as sns 

import math, random, ast

from os import listdir
from os.path import isfile, join



DATASET = "ColorsA"
EXP_DIR = "./trained_models/ColorsA/None/"
device = "cpu"
LOG_LEVELS = list(logging._levelToName.values())
log_level = "info"
NUM_EPOCHS = 1000

parser = argparse.ArgumentParser()
parser.add_argument('--rec_dist', type=str, default='gaussian')
parser.add_argument('--reg_anneal', type=int, default=10000)
parser.add_argument('--util_loss', type=str, default="mse")
parser.add_argument('--betaH_B', type=int, default=1)
parser.add_argument('--upsilon', type=int, default=1)
args = parser.parse_args()


formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(log_level.upper())
stream = logging.StreamHandler()
stream.setLevel(log_level.upper())
stream.setFormatter(formatter)
logger.addHandler(stream)

img_size = get_img_size("ColorsA")

model = init_specific_model("Burgess", "Malloy", img_size, 1)

model = model.to(device)  # make sure trainer and viz on same device
#gif_visualizer = GifTraversalsTraining(model, DATASET, EXP_DIR)
train_loader = get_dataloaders(DATASET,
                                batch_size=100,
                                logger=logger)

logger.info("Train {} with {} samples".format(DATASET, len(train_loader.dataset)))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_f = get_loss_f("betaH",
                    n_data=len(train_loader.dataset),
                    device=device,
                    **vars(args))

trainer = Trainer(model, optimizer, loss_f,
                    device=device,
                    logger=logger,
                    save_dir=EXP_DIR,
                    is_progress_bar=True)


utilities = np.zeros_like([1,1,1])
utilities = torch.from_numpy(utilities.astype(np.float64)).float()

trainer(train_loader,
        utilities=utilities, 
        epochs=NUM_EPOCHS,
        checkpoint_every=1000,)


for data in iter(train_loader):
    for idx, img in enumerate(data):
        print(img.shape)

        plt.imshow(img.permute(1, 2, 0).detach().numpy())
        #plt.show()
        plt.savefig('./trained_models/ColorsA/None/recon/original_stim_' + str(idx) + '.png')

        stim_1_recon , _, _, stim_1_pred_util = model(img.unsqueeze(0))

        stim_1_recon = stim_1_recon.squeeze()

        plt.imshow(stim_1_recon.permute(1, 2, 0).detach().numpy())
        #plt.show()
        plt.savefig('./trained_models/ColorsA/None/recon/recon_stim_' + str(idx) + '.png')

# SAVE MODEL AND EXPERIMENT INFORMATION
save_model(trainer.model, EXP_DIR, metadata=vars(args))

utilities = np.zeros_like([1,1,1])
utilities[0] = 1e6
utilities = torch.from_numpy(utilities.astype(np.float64)).float()

EXP_DIR = "./trained_models/ColorsA/Utility/"

trainer = Trainer(model, optimizer, loss_f,
                    device=device,
                    logger=logger,
                    save_dir=EXP_DIR,
                    is_progress_bar=True)

trainer(train_loader,
        utilities=utilities, 
        epochs=NUM_EPOCHS,
        checkpoint_every=1000,)


# SAVE MODEL AND EXPERIMENT INFORMATION
save_model(trainer.model, EXP_DIR, metadata=vars(args))

for data in iter(train_loader):
    for idx, img in enumerate(data):
        print(img.shape)

        stim_1_recon , _, _, stim_1_pred_util = model(img.unsqueeze(0))

        stim_1_recon = stim_1_recon.squeeze()

        plt.imshow(stim_1_recon.permute(1, 2, 0).detach().numpy())
        #plt.show()
        plt.savefig('./trained_models/ColorsA/Utility/recon/stim_' + str(idx) + '.png')
