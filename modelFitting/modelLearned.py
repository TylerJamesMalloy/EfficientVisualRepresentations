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
from matplotlib import pyplot as plt
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

    behaviorLearnedColumns = ["Id", "Number of Utility Observations", "Variance", "Utility", "Chosen"]
    behaviorLearned = pd.DataFrame([], columns=behaviorLearnedColumns)

    
    changes = []
    x0 = [0.0]

    good = []
    device = "cuda"

    POST_TRAIN = True 

    models = [[[],[],[]],
              [[],[],[]],
              [[],[],[]]]
    
    for uidx, upsilon in enumerate([4, 10, 100]):
        for bidx, Beta in enumerate(["Low", "Med", "High"]): 
            for stimuli_set in [0,1,2,3,4,5]:
                model = load_model("./trained_models/Marbles/BVAE/" + Beta + "/set" + str(stimuli_set))
                model.to(device)
                if(POST_TRAIN):
                    optimizer = optim.Adam(model.parameters(), lr=1e-6)

                    parser = argparse.ArgumentParser(description="",
                                            formatter_class=FormatterNoDuplicate)
                    parser.add_argument('--rec_dist', default="gaussian")
                    parser.add_argument('--reg_anneal', default=10000)
                    parser.add_argument('--util_loss', default="mse")
                    default_beta = 4
                    if(Beta == "High"):
                        default_beta = 100
                    if(Beta == "Med"):
                        default_beta = 10
                    parser.add_argument('--betaH_B', default=default_beta)
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

                    #utilities = np.array(stimuli_mean_utilities)
                    utilities = np.ones_like(stimuli_mean_utilities) * 2.5733 # mean utility
                    utilities = torch.from_numpy(utilities.astype(np.float64)).float().to(device)
                    trainer(train_loader,
                            utilities=utilities, 
                            epochs=2,
                            checkpoint_every=1e6,)

                models[bidx][uidx].append(model)

    
    
    for idx, id in enumerate(ids):        
        idf = df[df["Id"] == id]
        idf = idf.tail(180) # Skip learning trials
        #if(idx > 1): continue 
        print(idx)
        nlls = []
        accuracy = []
        inv_temp = 5
        set = int(idf['Marble Set'].unique()[0])
        participant_model = copy.deepcopy(models[2][2][set])

        all_marble_colors = pd.read_csv("./data/marbles/colors.csv")
        all_marble_colors = all_marble_colors['colors']

        stimuli = None 
        for _, stimuli in enumerate(train_loader):
            stimuli = stimuli

        stimuli_mean_utilities = []
        stimuli_deviations = []
        for marble_colors in all_marble_colors:
            marble_colors = np.array(ast.literal_eval(marble_colors))
            marble_values = np.select([marble_colors == 2, marble_colors == 1, marble_colors == 0], [2, 3, 4], marble_colors)
            stimuli_deviations.append(np.std(marble_values))
            stimuli_mean_utilities.append(np.mean(marble_values))

        old_block = 1.0 
        num_util_observations = 0

        for _, trial in idf.iterrows(): 
            if(trial['Block'] != old_block):
                old_block = trial['Block'] 
                num_util_observations = 0

            if(trial['Trial Type'] == 'Utility Selection'):
                num_util_observations += 1
                stim_1 = stimuli[int(trial['Left Stimulus'])].unsqueeze(0).to("cuda")
                stim_1_recon , _, _, stim_1_pred_util = participant_model(stim_1)

                stim_2 = stimuli[int(trial['Right Stimulus'])].unsqueeze(0).to("cuda")
                stim_2_recon , _, _, stim_2_pred_util = participant_model(stim_2)

                key_press = trial['Key Pressed']
                if(key_press != 'd' and key_press != 'f'):
                    break 

                left_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", trial['Left Stimulus Marbles'])) 
                rght_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", trial['Right Stimulus Marbles'])) 
                left_marbles = list(int(a) for a in left_marbles)
                rght_marbles = list(int(a) for a in rght_marbles)

                chosen_ev = np.mean(left_marbles) if trial['Key Pressed'] == "d" else np.mean(rght_marbles)
                chosen_var = np.var(left_marbles) if trial['Key Pressed'] == "d" else np.var(rght_marbles)

                unchosen_ev = np.mean(rght_marbles) if trial['Key Pressed'] == "d" else np.mean(left_marbles)
                unchosen_var = np.var(rght_marbles) if trial['Key Pressed'] == "d" else np.var(left_marbles)

                pred_utils = np.array([stim_1_pred_util.item(), stim_2_pred_util.item()])
                bvae_softmax = np.exp(pred_utils * inv_temp) / np.sum(np.exp(pred_utils * inv_temp), axis=0)

                utilities = np.array(stimuli_mean_utilities)
                utilities = torch.from_numpy(utilities.astype(np.float64)).float().to(device)
                trainer(train_loader,
                        utilities=utilities, 
                        epochs=2,
                        checkpoint_every=1e6,)

                if(key_press == 'd'):
                    nlls.append(np.log(bvae_softmax[0]))
                else:
                    nlls.append(np.log(bvae_softmax[1]))

                #behaviorLearnedColumns = ["Id", "Model", "Number of Utility Observations", "Variance", "Utility", "Chosen"]
                d = pd.DataFrame([[id, num_util_observations, chosen_var, chosen_ev, 1],
                                  [id, num_util_observations, unchosen_var, unchosen_ev, 0]], columns=behaviorLearnedColumns)
                behaviorLearned = pd.concat([behaviorLearned, d])

behaviorLearned.to_pickle('./behaviorLearned.pkl')

for observations in range(1,9):
    data = behaviorLearned[behaviorLearned["Number of Utility Observations"] == observations]
    grouped = data.groupby(['Variance']).mean()["Chosen"]

    regDataColumns = ["Variance", "Chosen"]
    regData = pd.DataFrame([], columns=regDataColumns)
    for key, value in grouped.items():
        d = pd.DataFrame([[key, value]], columns=regDataColumns) 
        regData = pd.concat([regData, d], ignore_index=True)

    r = stats.pearsonr(regData["Variance"], regData["Chosen"])
    print(r)



   