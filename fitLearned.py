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

def aversion(participant_model, stimuli, idf, test=False):
    nlls = []
    accuracy = []
    inv_temp = 5
    participant_model = copy.deepcopy(participant_model)

    all_marble_colors = pd.read_csv("./data/marbles/colors.csv")
    all_marble_colors = all_marble_colors['colors']

    stimuli_mean_utilities = []
    stimuli_deviations = []
    for marble_colors in all_marble_colors:
        marble_colors = np.array(ast.literal_eval(marble_colors))
        marble_values = np.select([marble_colors == 2, marble_colors == 1, marble_colors == 0], [2, 3, 4], marble_colors)
        stimuli_deviations.append(np.std(marble_values))
        stimuli_mean_utilities.append(np.mean(marble_values))

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
    
    del participant_model
    return np.mean(nlls)

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

    modelLearnedColumns = ["Id", "Model", "Split", "Log Likelihood", "Beta", "Upsilon"]
    modelLearned = pd.DataFrame([], columns=modelLearnedColumns)

    
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
        #if(idx > 10): continue 
        rng = np.random.default_rng(id)

        set = int(idf['Marble Set'].unique()[0])
        low_b_low_u = models[0][0][set]
        low_b_med_u = models[0][1][set]
        low_b_hgh_u = models[0][2][set]
        med_b_low_u = models[1][0][set]
        med_b_med_u = models[1][1][set]
        med_b_hgh_u = models[1][2][set]
        hgh_b_low_u = models[2][0][set]
        hgh_b_med_u = models[2][1][set]
        hgh_b_hgh_u = models[2][2][set]

        train_loader = get_dataloaders("Marbles",
                                        batch_size=100,
                                        logger=None,
                                        set=str(set))

        stimuli = None 
        for _, stimuli in enumerate(train_loader):
            stimuli = stimuli
        
        low_b_low_u_nll = aversion(low_b_low_u, stimuli, idf, True)
        low_b_med_u_nll = aversion(low_b_med_u, stimuli, idf, True)
        low_b_hgh_u_nll = aversion(low_b_hgh_u, stimuli, idf, True)
        med_b_low_u_nll = aversion(med_b_low_u, stimuli, idf, True)
        med_b_med_u_nll = aversion(med_b_med_u, stimuli, idf, True)
        med_b_hgh_u_nll = aversion(med_b_hgh_u, stimuli, idf, True)
        hgh_b_low_u_nll = aversion(hgh_b_low_u, stimuli, idf, True)
        hgh_b_med_u_nll = aversion(hgh_b_med_u, stimuli, idf, True)
        hgh_b_hgh_u_nll = aversion(hgh_b_hgh_u, stimuli, idf, True)

        coefs = [
            [4,4],
            [4,10],
            [4,100],
            [10,4],
            [10,10],
            [10,100],
            [100,4],
            [100,10],
            [100,100]
        ]

        results = [low_b_low_u_nll,
                    low_b_med_u_nll,
                    low_b_hgh_u_nll,
                    med_b_low_u_nll,
                    med_b_med_u_nll,
                    med_b_hgh_u_nll,
                    hgh_b_low_u_nll,
                    hgh_b_med_u_nll,
                    hgh_b_hgh_u_nll]

        best_nll = np.min(results)
        beta_coef = coefs[np.argmin(results)][0]
        upsilon_coef = coefs[np.argmin(results)][0]
        #modelLearnedColumns = ["Id", "Model", "Split", "Log Likelihood", "Coefficient"]
        d = pd.DataFrame([[id, "UBVAE", 100, best_nll, beta_coef, upsilon_coef]], columns=modelLearnedColumns)
        modelLearned = pd.concat([d, modelLearned])

        # 80% split 

        msk = rng.random(len(idf)) < 0.80
        train = idf[msk]

        ordered_models = [low_b_low_u,
                    low_b_med_u,
                    low_b_hgh_u,
                    med_b_low_u,
                    med_b_med_u,
                    med_b_hgh_u,
                    hgh_b_low_u,
                    hgh_b_med_u,
                    hgh_b_hgh_u]

        low_b_low_u_nll = aversion(low_b_low_u, stimuli, train, True)
        low_b_med_u_nll = aversion(low_b_med_u, stimuli, train, True)
        low_b_hgh_u_nll = aversion(low_b_hgh_u, stimuli, train, True)
        med_b_low_u_nll = aversion(med_b_low_u, stimuli, train, True)
        med_b_med_u_nll = aversion(med_b_med_u, stimuli, train, True)
        med_b_hgh_u_nll = aversion(med_b_hgh_u, stimuli, train, True)
        hgh_b_low_u_nll = aversion(hgh_b_low_u, stimuli, train, True)
        hgh_b_med_u_nll = aversion(hgh_b_med_u, stimuli, train, True)
        hgh_b_hgh_u_nll = aversion(hgh_b_hgh_u, stimuli, train, True)

        results = [low_b_low_u_nll,
                    low_b_med_u_nll,
                    low_b_hgh_u_nll,
                    med_b_low_u_nll,
                    med_b_med_u_nll,
                    med_b_hgh_u_nll,
                    hgh_b_low_u_nll,
                    hgh_b_med_u_nll,
                    hgh_b_hgh_u_nll]

        best_nll = np.min(results)

        test = idf[~msk]
        test_model = ordered_models[np.argmin(results)]

        test_nll = aversion(test_model, stimuli, test, True)

        beta_coef = coefs[np.argmin(results)][0]
        upsilon_coef = coefs[np.argmin(results)][0]
        #modelLearnedColumns = ["Id", "Model", "Split", "Log Likelihood", "Coefficient"]
        d = pd.DataFrame([[id, "UBVAE 80", 80, test_nll, beta_coef, upsilon_coef]], columns=modelLearnedColumns)
        modelLearned = pd.concat([d, modelLearned])

        # 60% split 

        msk = rng.random(len(idf)) < 0.60
        train = idf[msk]

        ordered_models = [low_b_low_u,
                    low_b_med_u,
                    low_b_hgh_u,
                    med_b_low_u,
                    med_b_med_u,
                    med_b_hgh_u,
                    hgh_b_low_u,
                    hgh_b_med_u,
                    hgh_b_hgh_u]

        low_b_low_u_nll = aversion(low_b_low_u, stimuli, train, True)
        low_b_med_u_nll = aversion(low_b_med_u, stimuli, train, True)
        low_b_hgh_u_nll = aversion(low_b_hgh_u, stimuli, train, True)
        med_b_low_u_nll = aversion(med_b_low_u, stimuli, train, True)
        med_b_med_u_nll = aversion(med_b_med_u, stimuli, train, True)
        med_b_hgh_u_nll = aversion(med_b_hgh_u, stimuli, train, True)
        hgh_b_low_u_nll = aversion(hgh_b_low_u, stimuli, train, True)
        hgh_b_med_u_nll = aversion(hgh_b_med_u, stimuli, train, True)
        hgh_b_hgh_u_nll = aversion(hgh_b_hgh_u, stimuli, train, True)

        results = [low_b_low_u_nll,
                    low_b_med_u_nll,
                    low_b_hgh_u_nll,
                    med_b_low_u_nll,
                    med_b_med_u_nll,
                    med_b_hgh_u_nll,
                    hgh_b_low_u_nll,
                    hgh_b_med_u_nll,
                    hgh_b_hgh_u_nll]

        best_nll = np.min(results)

        test = idf[~msk]
        test_model = ordered_models[np.argmin(results)]

        test_nll = aversion(test_model, stimuli, test, True)

        beta_coef = coefs[np.argmin(results)][0]
        upsilon_coef = coefs[np.argmin(results)][0]
        #modelLearnedColumns = ["Id", "Model", "Split", "Log Likelihood", "Coefficient"]
        d = pd.DataFrame([[id, "UBVAE 60", 60, test_nll, beta_coef, upsilon_coef]], columns=modelLearnedColumns)
        modelLearned = pd.concat([d, modelLearned])

        # 40% split 

        msk = rng.random(len(idf)) < 0.40
        train = idf[msk]

        ordered_models = [low_b_low_u,
                    low_b_med_u,
                    low_b_hgh_u,
                    med_b_low_u,
                    med_b_med_u,
                    med_b_hgh_u,
                    hgh_b_low_u,
                    hgh_b_med_u,
                    hgh_b_hgh_u]

        low_b_low_u_nll = aversion(low_b_low_u, stimuli, train, True)
        low_b_med_u_nll = aversion(low_b_med_u, stimuli, train, True)
        low_b_hgh_u_nll = aversion(low_b_hgh_u, stimuli, train, True)
        med_b_low_u_nll = aversion(med_b_low_u, stimuli, train, True)
        med_b_med_u_nll = aversion(med_b_med_u, stimuli, train, True)
        med_b_hgh_u_nll = aversion(med_b_hgh_u, stimuli, train, True)
        hgh_b_low_u_nll = aversion(hgh_b_low_u, stimuli, train, True)
        hgh_b_med_u_nll = aversion(hgh_b_med_u, stimuli, train, True)
        hgh_b_hgh_u_nll = aversion(hgh_b_hgh_u, stimuli, train, True)

        results = [low_b_low_u_nll,
                    low_b_med_u_nll,
                    low_b_hgh_u_nll,
                    med_b_low_u_nll,
                    med_b_med_u_nll,
                    med_b_hgh_u_nll,
                    hgh_b_low_u_nll,
                    hgh_b_med_u_nll,
                    hgh_b_hgh_u_nll]

        best_nll = np.min(results)

        test = idf[~msk]
        test_model = ordered_models[np.argmin(results)]

        test_nll = aversion(test_model, stimuli, test, True)

        beta_coef = coefs[np.argmin(results)][0]
        upsilon_coef = coefs[np.argmin(results)][0]
        #modelLearnedColumns = ["Id", "Model", "Split", "Log Likelihood", "Coefficient"]
        d = pd.DataFrame([[id, "UBVAE 40", 40, test_nll, beta_coef, upsilon_coef]], columns=modelLearnedColumns)
        modelLearned = pd.concat([d, modelLearned])


    modelLearned = modelLearned.reset_index()
    modelLearned.to_pickle("./modelLearned2.pkl")

    modelLearned = pd.read_pickle("./modelLearned2.pkl")
    modelLearned["Log Likelihood"] = -1 * modelLearned["Log Likelihood"] 

    # ANOVA
    result = stats.f_oneway(modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE"],
                            modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE 80"],
                            modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE 60"],
                            modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE 40"])
    print(result)

    res = stats.tukey_hsd(modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE"],
                            modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE 80"],
                            modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE 60"],
                            modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE 40"])
    print(res)
    

    sns.barplot(modelLearned, x="Model", y="Log Likelihood", hue="Split")
    plt.title("Learned")
    plt.show()
        
   