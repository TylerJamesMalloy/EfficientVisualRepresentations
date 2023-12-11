import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 




participantAversion = pd.read_pickle(".\participantAversion.pkl")
participantAversion = participantAversion.reset_index()
"""
axes[1].set_ylabel("Participant Count")
axes[1].set_xlabel("Risk Aversion Coefficient")
axes[1].set_title("Participant Risk Aversion Coefficients")"""

df = pd.read_csv("../data/participantData.csv")
"""'
Unnamed: 0', 'Id', 'Experiment Type', 'Marble Set', 'Block', 'Trial',
       'Trial Type', 'Reaction Time', 'Left Stimulus', 'Right Stimulus',
       'Left Stimulus Marbles', 'Right Stimulus Marbles', 'New Stimulus',
       'New Stimulus Marbles', 'Changed', 'Changed Index', 'Key Pressed',
       'Reward', 'Correct'
"""
df = df[df['Experiment Type'] == 'Decision-Making']

# 145 Decision Making Participants 

ids = df["Id"].unique()

#riskAversionColumns = ["Chosen EV", "Chosen Var", "Unchosen EV", "Unchosen Var"]
riskAversionColumns = ["Outcome Expected Value (Chosen - Unchosen)", "Outcome Variance (Chosen - Unchosen)"]
riskAversion = pd.DataFrame([], columns=riskAversionColumns)
riskAversionUnrounded = pd.DataFrame([], columns=riskAversionColumns)

selectionProbabilityColumns = ["Id", "Coefficient", "Utility Difference", "Variance Difference" , "Chosen"]
selectionProbability = pd.DataFrame([], columns=selectionProbabilityColumns)




for id in ids:
    idf = df[df["Id"] == id]
    idf = idf.tail(180) # Skip learning trials 

    if(len(participantAversion[participantAversion["Id"] == id]["Risk Aversion Coefficient"]) == 0): continue

    coefficient = participantAversion[participantAversion["Id"] == id]["Risk Aversion Coefficient"].item()
    for tindex, trial in idf.iterrows(): 
        if(trial['Trial Type'] == 'Utility Selection'):
            
            left_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", trial['Left Stimulus Marbles'])) 
            rght_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", trial['Right Stimulus Marbles'])) 
            left_marbles = list(int(a) for a in left_marbles)
            rght_marbles = list(int(a) for a in rght_marbles)
            # d = Left Stimulus, f = Right Stimulus
            chosen_ev = np.mean(left_marbles) if trial['Key Pressed'] == "d" else np.mean(rght_marbles)
            chosen_var = np.var(left_marbles) if trial['Key Pressed'] == "d" else np.var(rght_marbles)
            unchosen_ev = np.mean(rght_marbles) if trial['Key Pressed'] == "d" else np.mean(left_marbles)
            unchosen_var = np.var(rght_marbles) if trial['Key Pressed'] == "d" else np.var(left_marbles)

            if(np.isfinite(coefficient) and np.isfinite(chosen_ev) and np.isfinite(chosen_var) and np.isfinite(unchosen_ev) and np.isfinite(unchosen_var)):
                d = pd.DataFrame([[id, coefficient, chosen_ev - unchosen_ev, chosen_var - unchosen_var,  1]], columns=selectionProbabilityColumns)
                if(d.isna().any().any()): continue
                selectionProbability = pd.concat([selectionProbability, d], ignore_index=True)

                d = pd.DataFrame([[id, coefficient, unchosen_ev - chosen_ev , unchosen_var - chosen_var,  0]], columns=selectionProbabilityColumns)
                selectionProbability = pd.concat([selectionProbability, d], ignore_index=True)

selectionProbability["Chosen"] = pd.to_numeric(selectionProbability["Chosen"])  
selectionProbability["Id"] = pd.to_numeric(selectionProbability["Id"])
print(selectionProbability.dtypes)
#print(np.sum(selectionProbability["Utility Difference"] > 0))   # 6361: higher utility
#print(np.sum(selectionProbability["Utility Difference"] < 0))   # 4928: lower utility
#print(np.sum(selectionProbability["Utility Difference"] == 0))  # 1723: Equal utility 

fig, axes = plt.subplots(nrows=1, ncols=2)

# sns.regplot(modelBehaviorAversion_High, x="Outcome Variance Difference", y="Chosen", scatter=False, label="B=100", color="orange", ax=axes[0])
seeking = participantAversion[participantAversion['Risk Aversion Coefficient'] > 0]["Id"]
averse = participantAversion[participantAversion['Risk Aversion Coefficient'] < 0]["Id"]


low = selectionProbability[selectionProbability["Id"].isin(averse)]
high = selectionProbability[selectionProbability["Id"].isin(seeking)]

sns.regplot(data=high, x="Variance Difference", y="Chosen", scatter=False, ax=axes[0], color="Blue", label="Risk Averse Participants")
high["Variance Difference"] = high["Variance Difference"].round(1)
groups = high.groupby("Variance Difference").mean()
sns.scatterplot(data=groups, x="Variance Difference", y="Chosen", color="Blue", ax=axes[0])

sns.regplot(data=low, x="Variance Difference", y="Chosen", scatter=False, ax=axes[0], color="Orange", label="Risk Seeking Participants")
low["Variance Difference"] = low["Variance Difference"].round(1)
groups = low.groupby("Variance Difference").mean()
sns.scatterplot(data=groups, x="Variance Difference", y="Chosen", color="Orange", ax=axes[0])

axes[0].set_ylabel("Probability of Selection", fontsize = 14)
axes[0].set_xlabel("Outcome Variance Difference (Chosen - Unchosen)", fontsize = 14)
axes[0].set_title("Participant Selection by Outcome Variance Difference", fontsize = 16)

"""sns.histplot(data=selectionProbability, x="Utility Difference", bins=10, ax=axes[0])
group = selectionProbability.groupby("Utility Difference")["Chosen"].mean()
xd = group.index.values.tolist()
yd = group.values.tolist()

axes[0].set_ylabel("Count of Participant Choices", fontsize = 14)
axes[0].set_xlabel("Utility Difference (Chosen - Unchosen)", fontsize = 14)
axes[0].set_title(" Participant Choices by Utility Difference", fontsize = 16)"""

splitAversion = pd.read_pickle("./fitAversion.pkl")
splitAversion = splitAversion.reset_index()
splitAversion = splitAversion.astype({'Split': 'int32'})
splitAversion = splitAversion.rename(columns={"Log Likelihood": "Likelihood"})
splitAversion["Likelihood"] = -1 * splitAversion["Likelihood"] # scipy optimize minimizes the objective so it is made negative and switched here
EUT = splitAversion[splitAversion["Model"] == "EUT"]
splitAversion = splitAversion[(splitAversion["Likelihood"] < 7.5) & (splitAversion["Model"] == "CPT")] # Remove overly converged likelihoods (~1%) for CPT likelihood comparison
splitAversion = pd.concat([splitAversion, EUT])
#palette = sns.color_palette("mako", as_cmap=True)
#palette = sns.color_palette("flare", as_cmap=True)
palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, n_colors=4) # 

splitAversion.loc[splitAversion['Split'] == 20, 'Model'] = "CPT 40" # training label was used for this 
splitAversion.loc[splitAversion['Split'] == 40, 'Model'] = "CPT 60"
splitAversion.loc[splitAversion['Split'] == 60, 'Model'] = "CPT 80"


order = ["EUT", "CPT 40", "CPT 60", "CPT 80", "CPT"]

sns.barplot(data=splitAversion, x="Model", y="Likelihood", order=order,errorbar=('ci', 90), palette=palette, ax=axes[1], hue="Split")

axes[1].set_ylabel("Log Likelihood", fontsize = 14)
axes[1].set_xlabel("Model and Data Split", fontsize = 14)
axes[1].set_title("Log Likelihood by Model and Data Split", fontsize = 16)

splitAversion.to_pickle("../stats/participantAversion.pkl")

plt.show()


# Compare CPT with EUT
CPT_100_Likelihoods = splitAversion[splitAversion["Model"] == "CPT"]
CPT_100_Likelihoods = CPT_100_Likelihoods[CPT_100_Likelihoods["Split"] == 100]
CPT_100_Likelihoods = CPT_100_Likelihoods["Likelihood"]

EUT_Likelihoods = splitAversion[splitAversion["Model"] == "EUT"]
EUT_Likelihoods = EUT_Likelihoods["Likelihood"]

from scipy.stats import tukey_hsd
res = tukey_hsd(CPT_100_Likelihoods, EUT_Likelihoods)
print(res)


CPT_100_Likelihoods = splitAversion[splitAversion["Model"] == "CPT"]
CPT_100_Likelihoods = CPT_100_Likelihoods[CPT_100_Likelihoods["Split"] == 100]
CPT_100_Likelihoods = CPT_100_Likelihoods["Likelihood"]

CPT_20_Likelihoods = splitAversion[splitAversion["Split"] == 20]
CPT_20_Likelihoods = CPT_20_Likelihoods["Likelihood"]
res = tukey_hsd(CPT_20_Likelihoods, CPT_100_Likelihoods)
print(res)

CPT_40_Likelihoods = splitAversion[splitAversion["Split"] == 40]
CPT_40_Likelihoods = CPT_40_Likelihoods["Likelihood"]
res = tukey_hsd(CPT_40_Likelihoods, CPT_100_Likelihoods)
print(res)

CPT_60_Likelihoods = splitAversion[splitAversion["Split"] == 60]
CPT_60_Likelihoods = CPT_60_Likelihoods["Likelihood"]
res = tukey_hsd(CPT_60_Likelihoods, CPT_100_Likelihoods)
print(res)



"""
Another statistical test can be done using Wilks' lambda to compare the relationship between individual participants fit coefficients based on the split, which shows no statistical significance between individual participants λ parameters based on the split (df=325.9, F=0.1375, p=1.0). 
"""
"""
import os
import math 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from PIL import Image
from scipy.stats import f_oneway
from skimage.measure import shannon_entropy

fig, axes = plt.subplots(nrows=1, ncols=2)
splitAversion = pd.read_pickle(".\splitAversion_e1.pkl")
splitAversion = splitAversion.reset_index()





splitAversion["Log Likelihood"] = -1 * splitAversion["Log Likelihood"]
sns.barplot(splitAversion, x="Model", y="Log Likelihood", hue="Split", errorbar=('ci', 90), ax=axes[0])

# Statistic test to compare the coefficients across participants. 
plt.show()

assert(False)
fig, axes = plt.subplots(nrows=2, ncols=1)

learningAversion = pd.read_pickle(".\participantAversion_e2.pkl")
early = learningAversion[learningAversion["Type"] == "Early Trials"]
later = learningAversion[learningAversion["Type"] == "Later Trials"]

sns.histplot(early, x="Risk Aversion Coefficient", ax=axes[0])
sns.histplot(later, x="Risk Aversion Coefficient", ax=axes[1])

plt.show()

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

assert(False)
"""