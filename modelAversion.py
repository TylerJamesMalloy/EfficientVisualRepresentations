import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import argparse
from cmath import nan
import enum
import logging
from re import L
import sys
import os
import copy 
from configparser import ConfigParser
import scipy.stats as stats

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
from matplotlib import pyplot as plot
import seaborn as sns 

import math, random 

from os import listdir
from os.path import isfile, join

import ast 

if __name__ == '__main__':
    all_marble_colors = pd.read_csv("./data/marbles/colors.csv")
    all_marble_colors = all_marble_colors['colors']

    stimuli_mean_utilities = []
    stimuli_deviations = []
    for marble_colors in all_marble_colors:
        marble_colors = np.array(ast.literal_eval(marble_colors))
        marble_values = np.select([marble_colors == 2, marble_colors == 1, marble_colors == 0], [2, 3, 4], marble_colors)
        stimuli_deviations.append(np.std(marble_values))
        stimuli_mean_utilities.append(np.mean(marble_values))

    df = pd.read_csv("./data/participantData.csv")
    df = df[df['Experiment Type'] == 'Decision-Making']
    ids = df["Id"].unique()

    modelBehaviorAversionColumns = ["Model", "Outcome Variance Difference", "Outcome Utility Difference", "Beta", "Chosen"]
    modelBehaviorAversion = pd.DataFrame([], columns=modelBehaviorAversionColumns)
    
    changes = []
    x0 = [0.0]

    good = []
    device = "cuda"

    POST_TRAIN = False 

    models = []
    
    upsilon = 6
    beta =  6
    for stimuli_set in [0,1,2,3,4,5]:
        model = load_model("./trained_models/Marbles/BVAE/" + "Med" + "/set" + str(stimuli_set))
        model.to(device)
        if(POST_TRAIN):
            optimizer = optim.Adam(model.parameters(), lr=1e-6)

            parser = argparse.ArgumentParser(description="",
                                    formatter_class=FormatterNoDuplicate)
            parser.add_argument('--rec_dist', default="gaussian")
            parser.add_argument('--reg_anneal', default=10000)
            parser.add_argument('--util_loss', default="mse")
            parser.add_argument('--betaH_B', default=beta)
            parser.add_argument('--upsilon', default=upsilon)
            
            args = parser.parse_args()
            
            loss_f = get_loss_f("betaH",
                                n_data=100,
                                device=device,
                                **vars(args))
            trainer = Trainer(model, optimizer, loss_f,
                                device=device,
                                logger=None,
                                save_dir=None,
                                is_progress_bar=False)

            train_loader = get_dataloaders("Marbles",
                                        batch_size=100,
                                        logger=None,
                                        set=str(stimuli_set))

            utilities = np.array(stimuli_mean_utilities)
            utilities = torch.from_numpy(utilities.astype(np.float64)).float().to(device)
            trainer(train_loader,
                    utilities=utilities, 
                    epochs=1000,
                    checkpoint_every=1e6,)

        models.append(model)

    for idx, id in enumerate(ids):        
        idf = df[df["Id"] == id]
        idf = idf.tail(180) # Skip learning trials
        #if(idx > 10): continue 
        rng = np.random.default_rng(id)

        inv_temp = 5

    marble_set  = int(idf['Marble Set'].unique()[0])
    train_loader = get_dataloaders("Marbles",
                                        batch_size=100,
                                        logger=None,
                                        set=str(marble_set))

    participant_model = models[marble_set]
    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli

    for _, trial in idf.iterrows(): 
        if(trial['Trial Type'] == 'Utility Selection'):
            
            stim_1 = stimuli[int(trial['Left Stimulus'])].unsqueeze(0).to("cuda")
            stim_1_recon , _, _, stim_1_pred_util = participant_model(stim_1)

            stim_2 = stimuli[int(trial['Right Stimulus'])].unsqueeze(0).to("cuda")
            stim_2_recon , _, _, stim_2_pred_util = participant_model(stim_2)

            key_press = trial['Key Pressed']
            if(key_press != 'd' and key_press != 'f'):
                break 

            pred_utils = np.array([stim_1_pred_util.item(), stim_2_pred_util.item()])
            bvae_softmax = np.exp(pred_utils * inv_temp) / np.sum(np.exp(pred_utils * inv_temp), axis=0)

            left_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", trial['Left Stimulus Marbles'])) 
            rght_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", trial['Right Stimulus Marbles'])) 
            left_marbles = list(int(a) for a in left_marbles)
            rght_marbles = list(int(a) for a in rght_marbles)

            if(trial['Key Pressed'] != "d" and trial['Key Pressed'] != "f"): continue 
            chosen_ev = np.mean(left_marbles) if trial['Key Pressed'] == "d" else np.mean(rght_marbles)
            chosen_var = np.var(left_marbles) if trial['Key Pressed'] == "d" else np.var(rght_marbles)

            unchosen_ev = np.mean(rght_marbles) if trial['Key Pressed'] == "d" else np.mean(left_marbles)
            unchosen_var = np.var(rght_marbles) if trial['Key Pressed'] == "d" else np.var(left_marbles)

            #modelBehaviorAversionColumns = ["Model", "Outcome Variance Difference", "Outcome Utility Difference", "Beta", "Chosen"]
            

            d = pd.DataFrame([["UB-VAE", np.var(left_marbles) - np.var(rght_marbles),  np.mean(left_marbles) - np.mean(rght_marbles) , 100, bvae_softmax[0]]], columns=modelBehaviorAversionColumns)
            modelBehaviorAversion = pd.concat([d, modelBehaviorAversion])

            d = pd.DataFrame([["UB-VAE", np.var(rght_marbles) - np.var(left_marbles),  np.mean(rght_marbles) - np.mean(left_marbles) , 100, bvae_softmax[1]]], columns=modelBehaviorAversionColumns)
            modelBehaviorAversion = pd.concat([d, modelBehaviorAversion])



modelBehaviorAversion = modelBehaviorAversion.round(1)

group = modelBehaviorAversion.groupby(["Outcome Variance Difference"], as_index=False).agg({'Chosen':np.average})
group.reset_index(inplace=True)
sns.regplot(modelBehaviorAversion, x="Outcome Variance Difference", y="Chosen", scatter=False, label="B=100")
sns.scatterplot(group, x="Outcome Variance Difference", y="Chosen", label="B=100")
plot.show()

modelBehaviorAversion.to_pickle("modelBehaviorAversion_Med.pkl")
print(modelBehaviorAversion)      

   