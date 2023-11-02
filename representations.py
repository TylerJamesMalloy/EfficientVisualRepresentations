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

import math, random 

from os import listdir
from os.path import isfile, join

import ast 

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "trained_models/"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]


def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('name', type=str,
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'], choices=DATASETS)
    training.add_argument('-x', '--experiment',
                          default=default_config['experiment'], choices=EXPERIMENTS,
                          help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-ut', '--utility-type',
                       default=default_config['utility'], choices=UTILTIIES,
                       help='Type of utility prediction model to use.')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the latent variable.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")
    model.add_argument('-ul', '--util-loss',
                       default=default_config['util_loss'], choices=UTIL_LOSSES,
                       help="Type of Utility loss function to use.")
    model.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
                       choices=RECON_DIST,
                       help="Form of the likelihood ot use for each pixel.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

    # Loss Specific Options
    utility = parser.add_argument_group('BetaH specific parameters')
    utility.add_argument('-u', '--upsilon', type=float,
                       default=default_config['upsilon'],
                       help="Weight of the utility loss parameter.")
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B', type=float,
                       default=default_config['betaH_B'],
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC', type=float,
                       default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC', type=float,
                       default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-G', type=float,
                       default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--lr-disc', type=float,
                        default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')

    btcvae = parser.add_argument_group('beta-tcvae specific parameters')
    btcvae.add_argument('--btcvae-A', type=float,
                        default=default_config['btcvae_A'],
                        help="Weight of the MI term (alpha in the paper).")
    btcvae.add_argument('--btcvae-G', type=float,
                        default=default_config['btcvae_G'],
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    btcvae.add_argument('--btcvae-B', type=float,
                        default=default_config['btcvae_B'],
                        help="Weight of the TC term (beta in the paper).")

    # Eval options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')
    
    modelling = parser.add_argument_group('L&DM modelling specific options')
    modelling.add_argument('--model-epochs', type=int,
                            default=default_config['model_epochs'],
                            help='Number of epochs to train utility prediction model.')
    modelling.add_argument('--trial-update', type=str,
                            default=default_config['trial_update'],
                            help='Source for util predictions.')

    args = parser.parse_args(args_to_parse)
    if args.experiment != 'custom':
        if args.experiment not in ADDITIONAL_EXP:
            # update all common sections first
            model, dataset = args.experiment.split("_")
            common_data = get_config_section([CONFIG_FILE], "Common_{}".format(dataset))
            update_namespace_(args, common_data)
            common_model = get_config_section([CONFIG_FILE], "Common_{}".format(model))
            update_namespace_(args, common_model)

        try:
            experiments_config = get_config_section([CONFIG_FILE], args.experiment)
            update_namespace_(args, experiments_config)
        except KeyError as e:
            if args.experiment in ADDITIONAL_EXP:
                raise e  # only reraise if didn't use common section

    return args

all_marble_colors = pd.read_csv("./data/marbles/colors.csv")
all_marble_colors = all_marble_colors['colors']

stimuli_mean_utilities = []
stimuli_deviations = []
for marble_colors in all_marble_colors:
    marble_colors = np.array(ast.literal_eval(marble_colors))
    marble_values = np.select([marble_colors == 2, marble_colors == 1, marble_colors == 0], [2, 3, 4], marble_colors)
    stimuli_deviations.append(np.std(marble_values))
    stimuli_mean_utilities.append(np.mean(marble_values))

def log_loss_score(predicted, actual, eps=1e-14):
        """
        :param predicted:   The predicted probabilities as floats between 0-1
        :param actual:      The binary labels. Either 0 or 1.
        :param eps:         Log(0) is equal to infinity, so we need to offset our predicted values slightly by eps from 0 or 1
        :return:            The logarithmic loss between between the predicted probability assigned to the possible outcomes for item i, and the actual outcome.
        """
        predicted = np.clip(predicted, eps, 1-eps)
        loss = -1 * np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1-predicted))

        return loss

def softmax(utilities, inv_temp=10):
    print(utilities)
    soft_percentages = []
    for utility in utilities: 
        soft_percentages.append(np.exp(utility * inv_temp) / np.sum(np.exp(utilities * inv_temp), axis=0))
    
    return soft_percentages


LOW_BETA = 4   # Negative risk aversion coef
MED_BETA = 10  # Neutral risk aversion coef
HGH_BETA = 100 # Positive risk aversion coef

def main(args):
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    args.img_size = get_img_size(args.dataset)


    if(not os.path.exists(exp_dir + "/u0/")):
        os.makedirs(exp_dir + "/u0/")

    model = init_specific_model(args.model_type, args.utility_type, args.img_size, args.latent_dim)
    model = model.to(device)  # make sure trainer and viz on same device
    gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir + "/u0/")
    train_loader = get_dataloaders(args.dataset,
                                    batch_size=args.batch_size,
                                    logger=logger,
                                    set=0)
    
    logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_f = get_loss_f(args.loss,
                        n_data=len(train_loader.dataset),
                        device=device,
                        **vars(args))
    trainer = Trainer(model, optimizer, loss_f,
                        device=device,
                        logger=logger,
                        save_dir=exp_dir  + "/u0/",
                        is_progress_bar=not args.no_progress_bar,
                        gif_visualizer=gif_visualizer)
    
    utilities = np.zeros_like(stimuli_mean_utilities)
    utilities = torch.from_numpy(utilities.astype(np.float64)).float().to(device)
    trainer(train_loader,
            utilities=utilities, 
            epochs=args.epochs,
            checkpoint_every=args.checkpoint_every,)
    
    # Print model representations 

    marble_set   = 0
    train_loader = get_dataloaders("Marbles",
                                    batch_size=100,
                                    logger=None,
                                    set=str(marble_set))

    repColums = ["Utility", "Dimension 1", "Dimension 2", "Mean 1", "Mean 2", "Variance 1", "Variance 2"]
    reps = pd.DataFrame([[]], repColums)
    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli.to(device)
        stim_1_recons , stim_dists, stim_latents, stim_1_pred_utils = model(stimuli)
        stim_means = stim_dists[0].cpu().detach().numpy()
        stim_log_vars = stim_dists[1].cpu().detach().numpy()
        stim_1_pred_utils = stim_1_pred_utils.cpu().detach().numpy()
        stim_latents = stim_latents.cpu().detach().numpy()
        mean_util = np.mean(stim_1_pred_utils)

        for mean, log_var, latent, pred_util in zip(stim_means, stim_log_vars, stim_latents, stim_1_pred_utils):
            utility = "None"
            if(pred_util > mean_util):
                utility = "Low"
            else:
                utility = "High"

            d = pd.DataFrame([[utility, latent[0], latent[1], mean[0], mean[1], log_var[0], log_var[1]]], columns=repColums)
            reps = pd.concat([d, reps])
    
    reps = reps.dropna()
    sns.kdeplot(
        data=reps, x="Dimension 1", y="Dimension 2", hue="Utility",
        levels=5, thresh=.2,
    )

    reps.to_pickle("Representations.pkl")

    #plt.show()

    temp_model = copy.deepcopy(model)

    if(not os.path.exists(exp_dir + "/u100/")):
        os.makedirs(exp_dir + "/u100/")

    gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir + "/u100/")
    args.upsilon = 100
    loss_f = get_loss_f(args.loss,
                        n_data=len(train_loader.dataset),
                        device=device,
                        **vars(args))

    trainer = Trainer(temp_model, optimizer, loss_f,
                        device=device,
                        logger=logger,
                        save_dir=exp_dir +  "/u100/",
                        is_progress_bar=not args.no_progress_bar,
                        gif_visualizer=gif_visualizer)

    utilities = np.array(stimuli_mean_utilities)
    utilities = torch.from_numpy(utilities.astype(np.float64)).float().to(device)
    trainer(train_loader,
            utilities=utilities, 
            epochs=args.epochs,
            checkpoint_every=args.checkpoint_every,)
    
    # Print model representations 

    args.upsilon = 100
    loss_f = get_loss_f(args.loss,
                        n_data=len(train_loader.dataset),
                        device=device,
                        **vars(args))
    
    marble_set   = 0
    train_loader = get_dataloaders("Marbles",
                                    batch_size=100,
                                    logger=None,
                                    set=str(marble_set))

    repColums = ["Utility", "Dimension 1", "Dimension 2", "Mean 1", "Mean 2", "Variance 1", "Variance 2"]
    reps = pd.DataFrame([[]], repColums)
    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli.to(device)
        stim_1_recons , stim_dists, stim_latents, stim_1_pred_utils = temp_model(stimuli)
        stim_means = stim_dists[0].cpu().detach().numpy()
        stim_log_vars = stim_dists[1].cpu().detach().numpy()
        stim_1_pred_utils = stim_1_pred_utils.cpu().detach().numpy()
        stim_latents = stim_latents.cpu().detach().numpy()
        mean_util = np.mean(stim_1_pred_utils)

        for mean, log_var, latent, pred_util in zip(stim_means, stim_log_vars, stim_latents, stim_1_pred_utils):
            utility = "None"
            if(pred_util > mean_util):
                utility = "Low"
            else:
                utility = "High"

            d = pd.DataFrame([[utility, latent[0], latent[1], mean[0], mean[1], log_var[0], log_var[1]]], columns=repColums)
            reps = pd.concat([d, reps])
    
    reps = reps.dropna()
    sns.kdeplot(
        data=reps, x="Dimension 1", y="Dimension 2", hue="Utility",
        levels=5, thresh=.2,
    )

    reps.to_pickle("Representations_Trained_u100.pkl")

    temp_model = copy.deepcopy(model)

    if(not os.path.exists(exp_dir + "/u10/")):
        os.makedirs(exp_dir + "/u10/")

    gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir + "/u10/")
    args.upsilon = 10
    loss_f = get_loss_f(args.loss,
                        n_data=len(train_loader.dataset),
                        device=device,
                        **vars(args))

    trainer = Trainer(temp_model, optimizer, loss_f,
                        device=device,
                        logger=logger,
                        save_dir=exp_dir  + "/u10/",
                        is_progress_bar=not args.no_progress_bar,
                        gif_visualizer=gif_visualizer)

    utilities = np.array(stimuli_mean_utilities)
    utilities = torch.from_numpy(utilities.astype(np.float64)).float().to(device)
    trainer(train_loader,
            utilities=utilities, 
            epochs=args.epochs,
            checkpoint_every=args.checkpoint_every,)
    
    # Print model representations 
    
    marble_set   = 0
    train_loader = get_dataloaders("Marbles",
                                    batch_size=100,
                                    logger=None,
                                    set=str(marble_set))

    repColums = ["Utility", "Dimension 1", "Dimension 2", "Mean 1", "Mean 2", "Variance 1", "Variance 2"]
    reps = pd.DataFrame([[]], repColums)
    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli.to(device)
        stim_1_recons , stim_dists, stim_latents, stim_1_pred_utils = temp_model(stimuli)
        stim_means = stim_dists[0].cpu().detach().numpy()
        stim_log_vars = stim_dists[1].cpu().detach().numpy()
        stim_1_pred_utils = stim_1_pred_utils.cpu().detach().numpy()
        stim_latents = stim_latents.cpu().detach().numpy()
        mean_util = np.mean(stim_1_pred_utils)

        for mean, log_var, latent, pred_util in zip(stim_means, stim_log_vars, stim_latents, stim_1_pred_utils):
            utility = "None"
            if(pred_util > mean_util):
                utility = "Low"
            else:
                utility = "High"

            d = pd.DataFrame([[utility, latent[0], latent[1], mean[0], mean[1], log_var[0], log_var[1]]], columns=repColums)
            reps = pd.concat([d, reps])
    
    reps = reps.dropna()
    sns.kdeplot(
        data=reps, x="Dimension 1", y="Dimension 2", hue="Utility",
        levels=5, thresh=.2,
    )

    reps.to_pickle("Representations_Trained_u10.pkl")

    temp_model = copy.deepcopy(model)

    if(not os.path.exists(exp_dir + "/u1/")):
        os.makedirs(exp_dir + "/u1/")

    args.upsilon = 1
    gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir + "/u1/")
    loss_f = get_loss_f(args.loss,
                        n_data=len(train_loader.dataset),
                        device=device,
                        **vars(args))

    trainer = Trainer(temp_model, optimizer, loss_f,
                        device=device,
                        logger=logger,
                        save_dir=exp_dir  + "/u1/",
                        is_progress_bar=not args.no_progress_bar,
                        gif_visualizer=gif_visualizer)

    utilities = np.array(stimuli_mean_utilities)
    utilities = torch.from_numpy(utilities.astype(np.float64)).float().to(device)
    trainer(train_loader,
            utilities=utilities, 
            epochs=args.epochs,
            checkpoint_every=args.checkpoint_every,)
    
    # Print model representations 
    
    marble_set   = 0
    train_loader = get_dataloaders("Marbles",
                                    batch_size=100,
                                    logger=None,
                                    set=str(marble_set))

    repColums = ["Utility", "Dimension 1", "Dimension 2", "Mean 1", "Mean 2", "Variance 1", "Variance 2"]
    reps = pd.DataFrame([[]], repColums)
    stimuli = None 
    for _, stimuli in enumerate(train_loader):
        stimuli = stimuli.to(device)
        stim_1_recons , stim_dists, stim_latents, stim_1_pred_utils = temp_model(stimuli)
        stim_means = stim_dists[0].cpu().detach().numpy()
        stim_log_vars = stim_dists[1].cpu().detach().numpy()
        stim_1_pred_utils = stim_1_pred_utils.cpu().detach().numpy()
        stim_latents = stim_latents.cpu().detach().numpy()
        mean_util = np.mean(stim_1_pred_utils)

        for mean, log_var, latent, pred_util in zip(stim_means, stim_log_vars, stim_latents, stim_1_pred_utils):
            utility = "None"
            if(pred_util > mean_util):
                utility = "Low"
            else:
                utility = "High"

            d = pd.DataFrame([[utility, latent[0], latent[1], mean[0], mean[1], log_var[0], log_var[1]]], columns=repColums)
            reps = pd.concat([d, reps])
    
    reps = reps.dropna()
    sns.kdeplot(
        data=reps, x="Dimension 1", y="Dimension 2", hue="Utility",
        levels=5, thresh=.2,
    )

    reps.to_pickle("Representations_Trained_u1.pkl")

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)


