import os
import math 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from PIL import Image
from scipy.stats import f_oneway
from skimage.measure import shannon_entropy


fig, axes = plt.subplots(nrows=2, ncols=1)

learningAversion = pd.read_pickle(".\participantAversion_e2.pkl")
early = learningAversion[learningAversion["Type"] == "Early Trials"]
later = learningAversion[learningAversion["Type"] == "Later Trials"]

sns.histplot(early, x="Risk Aversion Coefficient", ax=axes[0])
sns.histplot(later, x="Risk Aversion Coefficient", ax=axes[1])

plt.show()
"""
directory = "../data/participantResponses"
df = pd.DataFrame()

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        data = pd.read_csv(os.path.join(directory, filename))
        data["Id"] = file.split("_")[0]

        df = pd.concat([data, df])

stimuliImages = np.asarray([np.load("../data/stimuli/stimuli" + str(x) + ".npy") for x in range(6)])
stimuliColors = pd.read_csv("../data/stimuli/colors.csv")
stimuliMeans = []
stimuliVars  = []
for stimuli in stimuliColors['colors']:
    stimuli = stimuli.replace("[", "")
    stimuli = stimuli.replace("]", "")
    stimuli = [int(s) for s in stimuli.split(',')]

    stimuli = [40 if x == 2 else x for x in stimuli]
    stimuli = [30 if x == 1 else x for x in stimuli]
    stimuli = [20 if x == 0 else x for x in stimuli]

    stimuliMeans.append(np.mean(stimuli))
    stimuliVars.append(np.var(stimuli))

participants = df["Id"].unique()

utilityBiasColumns = ["Id", "Chosen Utility Difference", "Lower Utility Option Chosen", "Visual Difference", "Utility Feedback"]
utilityBias = pd.DataFrame([], columns=utilityBiasColumns)

changeBiasColumns = ["Id", "Utility Difference", "Detected Change", "Utility Feedback"]
changeBias = pd.DataFrame([], columns=changeBiasColumns)

fig, ax = plt.subplots(nrows=1, ncols=2)

for participant in participants:
    pdf = df[df["Id"] == participant]
    pdf = pdf.tail(200) 
    pdf = pdf[pdf["reward"] != - 1.0]
    utilityTrials = pdf[pdf["type"] == 1.0]
    utilityFeedback = 0
    block = 0.0 

    marble_set = int(pdf["marble_set"].unique()[0])
    images = stimuliImages[marble_set, :, :, :]

    for index, row in pdf.iterrows():
        if(math.isnan(row['block'])): continue
        if(block != row['block']):
            utilityFeedback = 0
            block = row['block']
        
        if(row["type"] == 1.0):
            stim_1 = images[int(row["stim_1"]), :, :, :]
            stim_2 = images[int(row["stim_2"]), :, :, :]

            stim_1_util = stimuliMeans[int(row["stim_1"])]
            stim_2_util = stimuliMeans[int(row["stim_2"])]

            visualDiff = np.sqrt(np.sum((stim_1 - stim_2)^2)) / stim_1.size
            utilityDiff = stim_1_util - stim_2_util if row["key_press"] == "d" else stim_2_util - stim_1_util

            lowerUtilityOptionChosen = stim_1_util <= stim_2_util if row["key_press"] == "d" else stim_2_util <= stim_1_util
            ub = pd.DataFrame([[participant, utilityDiff, float(lowerUtilityOptionChosen), visualDiff, utilityFeedback]], columns=utilityBiasColumns)
            utilityBias = pd.concat([utilityBias, ub])

            utilityFeedback += 1
        
        if(row["type"] == 0.0):
            if(row["changed"] == 1.0):
                if(row["reward"] == 30.0):
                    detected_change = 1.0
                else:
                    detected_change = 0.0
                stims = [int(row["stim_1"]), int(row["stim_2"])]
                change_index = int(row["change_index"])
                changed_stim = stims[change_index]

                new_stim_util = stimuliMeans[int(row["new_stim"])]
                changed_stim_util = stimuliMeans[changed_stim]
                utilityDiff = (np.square(new_stim_util - changed_stim_util)).mean() / 125
                
                cb = pd.DataFrame([[participant, utilityDiff, detected_change, utilityFeedback]], columns=changeBiasColumns)
                changeBias = pd.concat([changeBias, cb])
                




ax[0].set_ylim(0.75, 1.25)
ax[0].legend()
ax[0].set_ylabel("Change Detection Bias", fontsize=14)
ax[0].set_xlabel("Utility Difference", fontsize=14)
ax[0].set_title("Utility Selection Bias by Visual Difference", fontsize=18)

ax[1].set_ylim(0.55, 0.65)
ax[1].legend()
ax[1].set_ylabel("Utility Selection Bias", fontsize=14)
ax[1].set_xlabel("Visual Difference", fontsize=14)
ax[1].set_title("Change Detection Bias by Utility Difference", fontsize=18)
plt.suptitle("Participant Biases by Early and Late Trials", fontsize=18)
plt.show()

assert(False)"""