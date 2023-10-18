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

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

DATASET = "ColorsA"
EXP_DIR = "./trained_models/ColorsB/None/"
device = "cpu"
LOG_LEVELS = list(logging._levelToName.values())
log_level = "info"
NUM_EPOCHS = 100
TRAINA = 0 # static utility
TRAINB = 0 # variable utility

TESTA = 0 # static utility
TESTB = 0 # variable utility

PLOTA = 0
PLOTB = 1

parser = argparse.ArgumentParser()
parser.add_argument('--rec_dist', type=str, default='gaussian')
parser.add_argument('--reg_anneal', type=int, default=10000)
parser.add_argument('--util_loss', type=str, default="mse")
parser.add_argument('--betaH_B', type=int, default=1)
parser.add_argument('--upsilon', type=int, default=50)
parser.add_argument('--img_size', type=list, default=[3,64,64])
parser.add_argument('--latent_dim', type=int, default=1)
parser.add_argument('--model_type', type=str, default="Burgess")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dataset', type=str, default="ColorsA")
parser.add_argument('--utility_type', type=str, default="Malloy")

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

if(TRAINB):
        logger.info("Train {} with {} samples".format(DATASET, len(train_loader.dataset)))
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        loss_f = get_loss_f("betaH",
                        n_data=len(train_loader.dataset),
                        device=device,
                        **vars(args))

        trainer = Trainer(model, optimizer, loss_f,
                        device=device,
                        logger=logger,
                        save_dir="./trained_models/ColorsB/None/",
                        is_progress_bar=True)


        #utilities = np.zeros_like([1,1,1])
        utilities = np.array([.333,.333,.333])
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
                        plt.savefig('./trained_models/ColorsB/None/recon/original_stim_' + str(idx) + '.png')

                        stim_1_recon , _, _, stim_1_pred_util = model(img.unsqueeze(0))

                        stim_1_recon = stim_1_recon.squeeze()

                        plt.imshow(stim_1_recon.permute(1, 2, 0).detach().numpy())
                        #plt.show()
                        plt.savefig('./trained_models/ColorsB/None/recon/recon_stim_' + str(idx) + '.png')

        # SAVE MODEL AND EXPERIMENT INFORMATION
        save_model(trainer.model, "./trained_models/ColorsB/None/", metadata=vars(args))

        #utilities = np.zeros_like([1,1,1])
        
        for _ in range(NUM_EPOCHS):
                #logger.info("Train {} with {} samples".format(DATASET, len(train_loader.dataset)))

                u0 = np.random.normal(0.5 , 0.1, 1)[0]
                u1 = np.random.normal(0.5 , 0.1, 1)[0]
                u2 = np.random.normal(0.75, 0.5, 1)[0]

                utilities = np.array([u0, u1, u2])
                utilities = torch.from_numpy(utilities.astype(np.float64)).float()

                EXP_DIR = "./trained_models/ColorsB/Utility/"

                trainer = Trainer(model, optimizer, loss_f,
                                device=device,
                                logger=None,
                                save_dir=EXP_DIR,
                                is_progress_bar=True)

                trainer(train_loader,
                        utilities=utilities, 
                        epochs=1,
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
                        print("stim: ", str(idx))
                        print(stim_1_pred_util)
                        plt.savefig('./trained_models/ColorsB/Utility/recon/stim_' + str(idx) + '.png')


if(TRAINA):
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


        #utilities = np.zeros_like([1,1,1])
        utilities = np.array([.333,.333,.333])
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

        #utilities = np.zeros_like([1,1,1])
        utilities = np.array([0,0,1])
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
                        print("stim: ", str(idx))
                        print(stim_1_pred_util)
                        plt.savefig('./trained_models/ColorsA/Utility/recon/stim_' + str(idx) + '.png')

def my_gauss(x, sigma=1, h=1, mid=0):
    from math import exp, pow
    variance = pow(sigma, 2)
    return h * exp(-pow(x-mid, 2)/(2*variance))

import scipy.stats as stats
from sklearn.metrics import mean_squared_error

if(TESTA):
    plt.cla()
    plt.clf()

    model = load_model("./trained_models/ColorsA/None/")

    for data in iter(train_loader):
        dists = []
        recon_errors = []
        utility_errors = []
        for idx, img in enumerate(data):
                stim_1_recon , stim_1_dist, stim_1_sample, stim_1_pred_util = model(img.unsqueeze(0))
                mu = stim_1_dist[0].cpu().detach().numpy()[0]
                sigma = stim_1_dist[1].cpu().detach().numpy()[0]
                sigma = math.e ** sigma
                sigma = np.sqrt(sigma) * 2
                dists.append([mu, sigma])

                original = img.cpu().detach().numpy()
                recon = stim_1_recon[0].cpu().detach().numpy()
                recon_error = np.sum(np.sqrt((original-recon)**2))
                recon_errors.append(recon_error)

                pred_util = stim_1_pred_util.cpu().detach().numpy()[0]
                util_error = 100 * ((pred_util - (1/3)) ** 2)
                utility_errors.append(util_error)
        
        
        print(np.mean(recon_errors)) # 41.84
        print(np.mean(utility_errors)) # 3.821 \times 10^{-3}

        colors = [(0.75,0.5,0.0), (0.5, 0.75, 0), (0.5, 0, 0.75)]
        for idx, dist in enumerate(dists):
                color = colors[idx]
                #value = np.random.normal(loc=dist[0],scale=dist[1],size=(1, 1000))
                #sns.distplot(value)
                print(dist[0])
                x = np.linspace(dist[0] - 3*dist[1], dist[0] + 3*dist[1], 100)
                plt.plot(x, stats.norm.pdf(x, dist[0], dist[1]), color = color)
        
        plt.show()

        model = load_model("./trained_models/ColorsA/Utility/")

    for data in iter(train_loader):
        dists = []
        recon_errors = []
        utility_errors = []
        utils = [0,0,1]
        for idx, img in enumerate(data):
                stim_1_recon , stim_1_dist, stim_1_sample, stim_1_pred_util = model(img.unsqueeze(0))

                print(stim_1_dist)
                util = utils[idx]

                mu = stim_1_dist[0].cpu().detach().numpy()[0]
                sigma = stim_1_dist[1].cpu().detach().numpy()[0]
                sigma = math.e ** sigma
                sigma = np.sqrt(sigma) * 2
                dists.append([mu, sigma])

                original = img.cpu().detach().numpy()
                recon = stim_1_recon[0].cpu().detach().numpy()
                recon_error = np.sum(np.sqrt((original-recon)**2))
                recon_errors.append(recon_error)

                pred_util = stim_1_pred_util.cpu().detach().numpy()[0]
                util_error = 100 * ((pred_util - (util)) ** 2)
                utility_errors.append(util_error)
        
        
        print(np.mean(recon_errors)) # 48.36
        print(np.mean(utility_errors)) # 2.37 \times 10^{-2}
        print(dists)
        colors = [(0.75,0.5,0.0), (0.5, 0.75, 0), (0.5, 0, 0.75)]
        for idx, dist in enumerate(dists):
                color = colors[idx]
                #value = np.random.normal(loc=dist[0],scale=dist[1],size=(1, 1000))
                #sns.distplot(value)
                print(dist[0])
                x = np.linspace(dist[0] - 3*dist[1], dist[0] + 3*dist[1], 100)
                plt.plot(x, stats.norm.pdf(x, dist[0], dist[1]), color = color)

                
        
        plt.show()

if(TESTB):
    plt.cla()
    plt.clf()

    model = load_model("./trained_models/ColorsB/None/")

    for data in iter(train_loader):
        dists = []
        recon_errors = []
        utility_errors = []
        predicted_utilities = []
        for idx, img in enumerate(data):
                stim_1_recon , stim_1_dist, stim_1_sample, stim_1_pred_util = model(img.unsqueeze(0))
                mu = stim_1_dist[0].cpu().detach().numpy()[0]
                sigma = stim_1_dist[1].cpu().detach().numpy()[0]
                sigma = math.e ** sigma
                sigma = np.sqrt(sigma) * 2
                dists.append([mu, sigma])

                original = img.cpu().detach().numpy()
                recon = stim_1_recon[0].cpu().detach().numpy()
                recon_error = np.sum(np.sqrt((original-recon)**2))
                recon_errors.append(recon_error)

                pred_util = stim_1_pred_util.cpu().detach().numpy()[0]
                util_error = 100 * ((pred_util - (1/3)) ** 2)
                utility_errors.append(util_error)
        
        print(recon_errors)
        assert(False)
        print(np.mean(recon_errors)) # 41.84
        print(np.mean(utility_errors)) # 3.821 \times 10^{-3}

        colors = [(0.75,0.5,0.0), (0.5, 0.75, 0), (0.5, 0, 0.75)]
        for idx, dist in enumerate(dists):
                color = colors[idx]
                #value = np.random.normal(loc=dist[0],scale=dist[1],size=(1, 1000))
                #sns.distplot(value)
                print(dist[0])
                x = np.linspace(dist[0] - 3*dist[1], dist[0] + 3*dist[1], 100)
                plt.plot(x, stats.norm.pdf(x, dist[0], dist[1]), color = color)
        
        plt.show()

        model = load_model("./trained_models/ColorsB/Utility/")

    for data in iter(train_loader):
        dists = []
        recon_errors = []
        utility_errors = []
        utils = [0,0,1]
        for idx, img in enumerate(data):
                stim_1_recon , stim_1_dist, stim_1_sample, stim_1_pred_util = model(img.unsqueeze(0))

                print(stim_1_dist)
                util = utils[idx]

                mu = stim_1_dist[0].cpu().detach().numpy()[0]
                sigma = stim_1_dist[1].cpu().detach().numpy()[0]
                sigma = math.e ** sigma
                sigma = np.sqrt(sigma) * 2
                dists.append([mu, sigma])

                original = img.cpu().detach().numpy()
                recon = stim_1_recon[0].cpu().detach().numpy()
                recon_error = np.sum(np.sqrt((original-recon)**2))
                recon_errors.append(recon_error)

                pred_util = stim_1_pred_util.cpu().detach().numpy()[0]
                util_error = 100 * ((pred_util - (util)) ** 2)
                utility_errors.append(util_error)
        
        
        print(np.mean(recon_errors)) # 48.36
        print(np.mean(utility_errors)) # 2.37 \times 10^{-2}
        print(dists)
        colors = [(0.75,0.5,0.0), (0.5, 0.75, 0), (0.5, 0, 0.75)]
        for idx, dist in enumerate(dists):
                color = colors[idx]
                #value = np.random.normal(loc=dist[0],scale=dist[1],size=(1, 1000))
                #sns.distplot(value)
                print(dist[0])
                x = np.linspace(dist[0] - 3*dist[1], dist[0] + 3*dist[1], 100)
                plt.plot(x, stats.norm.pdf(x, dist[0], dist[1]), color = color)

                
        
        plt.show()


from scipy.stats import norm

if(PLOTB):
       print("in plot b")

       fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True)
       # High Beta Value
       mean_1 = np.array([0.6, 0.6, 0.6])
       std_1 = np.array([.01, .01, .01])

       # Low Beta Value
       mean_2 = np.array([0.6, 0.6, 0.6])
       std_2 = np.array([.01, .01, .01])
       
       x = [0.1, 0.1, 0.5]
       axes[0,1].plot(x, mean_1, 'b-', label=r'$\beta = 1$')
       axes[0,1].fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
       axes[0,1].plot(x, mean_2, 'r-', label=r'$\beta = 100$')
       axes[0,1].fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
       axes[0,1].legend()

       axes[0,1].set_title(r'$\beta$ ' + " Variational Auto Encoder")

       # High Beta Value
       mean_1 = np.array([0.4482, 0.4573, 0.7023])
       std_1 = np.array([.01, .01, .01])

       # Low Beta Value
       mean_2 = np.array([0.5918, 0.5917, 0.4304])
       std_2 = np.array([.01, .01, .01])
       
       x = [0.1, 0.1, 0.5]
       axes[0,0].plot(x, mean_1, 'b-', label=r'$\beta = 1$')
       axes[0,0].fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
       axes[0,0].plot(x, mean_2, 'r-', label=r'$\beta = 100$')
       axes[0,0].fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
       axes[0,0].legend()

       axes[0,0].set_ylabel("Marble Utility " + r'$u(x)$')

       axes[0,0].set_title("Utility " + r'$\beta$ ' + " Variational Auto Encoder")

       mean_1 = np.array([0.6, 0.6, 0.6])
       std_1 = np.array([.01, .01, .01])

       # Low Beta Value
       mean_2 = np.array([0.6, 0.6, 0.6])
       std_2 = np.array([.01, .01, .01])
       
       x = [0.1, 0.1, 0.5]
       axes[1,0].plot(x, mean_1, 'b-', label="VAE")
       axes[1,0].fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
       axes[1,0].set_title("Utility Variational Auto Encoder")
       axes[1,0].legend()

       axes[1,0].set_xlabel("Utility Prediction Error " + r'$\upsilon(U(z)-u(x))$')
       axes[1,0].set_ylabel("Marble Utility " + r'$u(x)$')

       mean_1 = np.array([0.6, 0.6, 0.6])
       std_1 = np.array([.01, .01, .01])

       # Low Beta Value
       mean_2 = np.array([0.6, 0.6, 0.6])
       std_2 = np.array([.01, .01, .01])
       
       x = [0.1, 0.1, 0.5]
       axes[1,1].plot(x, mean_1, 'b-', label="VAE")
       axes[1,1].fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
       axes[1,1].legend()

       axes[1,1].set_title("Variational Auto Encoder")
       axes[1,1].set_xlabel("Utility Prediction Error " + r'$\upsilon(U(z)-u(x))$')

       plt.suptitle("Utility Prediction Error by Marble Utility")
        
       plt.show()

if(PLOTA):
        #plt.rcParams['text.usetex'] = True
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)

        axes[0].set_xlabel("Representation Value (Z)")
        axes[1].set_xlabel("Representation Value (Z)")

        axes[0].set_ylabel("Representation Probaiblity Density (Z)")

        axes[0].set_title("Before Utility Training")
        axes[1].set_title("After Utility Training")

        dists = [[-2.1744, 0.23192586] , [-0.9228, 0.23192586], [0.2437, 0.23192586]]
        colors = [(0.75,0.5,0.0), (0.5, 0.75, 0), (0.5, 0, 0.75)]
        color_labels = ["Orange Marble", "Green Marble", "Purple Marble"]
        for idx, dist in enumerate(dists):
                color = colors[idx]
                color_label = color_labels[idx]
                x = np.linspace(dist[0] - 3*dist[1], dist[0] + 3*dist[1], 100)
                axes[0].plot(x, stats.norm.pdf(x, dist[0], dist[1]), color = color, label = color_label)
                axes[0].legend()
        
        kl_divergences = []
        for dist1 in dists:
               for dist2 in dists:
                        if(all(x == y for x, y in zip(dist1, dist2))): continue 

                        pdf1 = norm.pdf(x, loc=dist1[0], scale=dist1[1])
                        pdf2 = norm.pdf(x, loc=dist2[0], scale=dist2[1])
                        kl_divergence = np.sum(np.where(pdf1 != 0, pdf1 * np.log(pdf1 / pdf2), 0))
                        kl_divergences.append(kl_divergence)
        
        mean_kl = round(np.mean(kl_divergences), 2)
        Recon_error = r'$E_{(q_{\phi} (z│x))}[log⁡[p_{\theta} (x|z)]] = 41.84$'
        axes[0].text(0.05, 0.7, Recon_error, transform=axes[0].transAxes, fontsize=12)

        KLD = r'$\beta D_{KL} (q_{\phi} (z│x)|p(z)) = 800.09$'
        axes[0].text(0.05, 0.625, KLD, transform=axes[0].transAxes, fontsize=12)

        Util_error = r'$\upsilon(U(z)-u(x))^2 = 3.82 \times 10^{-3}$'
        axes[0].text(0.05, 0.55, Util_error, transform=axes[0].transAxes, fontsize=12)
        
        dists = [[-2.1258, 0.2192586] , [-1.4379, 0.2192586], [-0.1543, 0.09355588 ]]
        colors = [(0.75,0.5,0.0), (0.5, 0.75, 0), (0.5, 0, 0.75)]
        for idx, dist in enumerate(dists):
                color = colors[idx]
                color_label = color_labels[idx]
                x = np.linspace(dist[0] - 3*dist[1], dist[0] + 3*dist[1], 100)
                axes[1].plot(x, stats.norm.pdf(x, dist[0], dist[1]), color = color, label = color_label)
                axes[1].legend()
        

        kl_divergences = []
        for dist1 in dists:
               for dist2 in dists:
                        if(all(x == y for x, y in zip(dist1, dist2))): continue 

                        pdf1 = norm.pdf(x, loc=dist1[0], scale=dist1[1])
                        pdf2 = norm.pdf(x, loc=dist2[0], scale=dist2[1])
                        kl_divergence = np.sum(np.where(pdf1 != 0, pdf1 * np.log(pdf1 / pdf2), 0))
                        kl_divergences.append(kl_divergence)
                        #print("KL DIV:", kl_divergence)
        
        
        Recon_error = r'$E_{(q_{\phi} (z│x))}[log⁡[p_{\theta} (x|z)]] = 48.36$'
        axes[1].text(0.05, 0.7, Recon_error, transform=axes[1].transAxes, fontsize=12)
        
        mean_kl = round(np.mean(kl_divergences), 2)
        print(mean_kl)
        KLD = r'$\beta D_{KL} (q_{\phi} (z│x)|p(z)) = 1714.39$'
        axes[1].text(0.05, 0.625, KLD, transform=axes[1].transAxes, fontsize=12)

        Util_error = r'$υ(U(z)-u(x))^2 = 2.37 \times 10^{-2}$'
        axes[1].text(0.05, 0.55, Util_error, transform=axes[1].transAxes, fontsize=12)

        fig.suptitle("Marble Stimuli Representation Distributions \n Before and After Utility Training")
        # Add KL Divergence 
        plt.show()

        
