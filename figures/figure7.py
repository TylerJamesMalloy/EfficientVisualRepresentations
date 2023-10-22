import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr 

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
changeDetectionColumns = ["Id", "Visual Difference", "Utility Difference", "Detected"]
changeDetection = pd.DataFrame([], columns=changeDetectionColumns)

for id in ids:
    idf = df[df["Id"] == id]
    idf = idf.tail(180) # Skip learning trials 
    for _, trial in idf.iterrows(): 
        if(trial['Trial Type'] == 'Change Detection' and trial['Changed']):
            
            original_marbles = trial['Right Stimulus Marbles'] if trial['Changed Index'] else trial['Left Stimulus Marbles']
            new_marbles = trial['New Stimulus Marbles'] 
            original_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", original_marbles)) 
            new_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", new_marbles)) 
            original_marbles = list(int(a) for a in original_marbles)
            new_marbles = list(int(a) for a in new_marbles)

            original_ev = np.mean(original_marbles)
            new_ev = np.mean(new_marbles)

            max_diff = (2 - 4) ** 2
            ev_diff = ((original_ev - new_ev) ** 2) / max_diff
            vis_diff = np.sum([original_marbles[x] != new_marbles[x] for x in range(len(original_marbles))]) / 9
            correct = trial["Correct"]

            d = pd.DataFrame([[id, vis_diff, ev_diff, correct]], columns=changeDetectionColumns)
            changeDetection = pd.concat([changeDetection, d])        


grouped = changeDetection.groupby(['Id']).mean()["Detected"]
grouped = grouped[grouped > 0.55]
good = grouped.keys().tolist()
changeDetection = changeDetection[changeDetection["Id"].isin(good)]

x = changeDetection["Utility Difference"].unique()
grouped = changeDetection.groupby(['Utility Difference']).mean()["Detected"]

regDataColumns = ["Utility Difference", "Detection Probability"]
regData = pd.DataFrame([], columns=regDataColumns)
for key, value in grouped.items():
    d = pd.DataFrame([[key, value]], columns=regDataColumns) 
    regData = pd.concat([regData, d], ignore_index=True)

regData = regData.reset_index()

fig, axes = plt.subplots(nrows=1, ncols=3)
sns.regplot(data=regData, x="Utility Difference", y="Detection Probability", ax=axes[0])

r = pearsonr(regData["Utility Difference"], regData["Detection Probability"])
print(r) # (statistic=0.87, pvalue=1.0E-5)
axes[0].text(0.05, 0.8, "R=0.87", transform=axes[0].transAxes, fontsize=12)
axes[0].text(0.05, 0.75, "P=1.0E-5", transform=axes[0].transAxes, fontsize=12)

grouped = changeDetection.groupby(['Visual Difference']).mean()["Detected"]

regDataColumns = ["Visual Difference", "Detection Probability"]
regData = pd.DataFrame([], columns=regDataColumns)
for key, value in grouped.items():
    if(key == 0): continue
    d = pd.DataFrame([[key, value]], columns=regDataColumns) 
    regData = pd.concat([regData, d], ignore_index=True)

regData = regData.reset_index()
sns.regplot(data=regData, x="Visual Difference", y="Detection Probability", ax=axes[1])

r = pearsonr(regData["Visual Difference"], regData["Detection Probability"])
print(r) # (statistic=0.3848826323868424, pvalue=0.30637763494591747)
axes[1].text(0.05, 0.8, "R=0.38", transform=axes[1].transAxes, fontsize=12)
axes[1].text(0.05, 0.75, "P=0.30", transform=axes[1].transAxes, fontsize=12)

splitChange = pd.read_pickle(".\splitChange.pkl")
splitChange = splitChange.reset_index()
splitChange.loc[splitChange['Split'] == 80, 'Model'] = "Utility 60"
splitChange.loc[splitChange['Split'] == 60, 'Model'] = "Utility 80"
splitChange.loc[splitChange['Split'] == 40, 'Model'] = "Utility 40"

print(splitChange['Split'].unique())

Utility = splitChange[splitChange['Model'] == "Utility"]
print(Utility)
import scipy.stats as stats

result = stats.f_oneway(splitChange['Likelihood'][splitChange['Model'] == "Utility"],
                        splitChange['Likelihood'][splitChange['Model'] == "Utility 80"],
                        splitChange['Likelihood'][splitChange['Model'] == "Utility 60"],
                        splitChange['Likelihood'][splitChange['Model'] == "Utility 40"],
                        splitChange['Likelihood'][splitChange['Model'] == "Visual"])

print(result)
assert(False)




assert(False)

order = ["Visual", "Utility 40", "Utility 60", "Utility 80", "Utility"]

splitChange.loc[splitChange['Split'] == 80, 'Model'] = "Utility 60"
splitChange.loc[splitChange['Split'] == 60, 'Model'] = "Utility 80"
splitChange.loc[splitChange['Split'] == 40, 'Model'] = "Utility 40"

Utility_100_Likelihoods = splitChange[splitChange["Model"] == "Utility"]
Utility_100_Likelihoods = Utility_100_Likelihoods[Utility_100_Likelihoods["Split"] == 100]
Utility_100_Likelihoods = Utility_100_Likelihoods["Likelihood"]

Visual_Likelihoods = splitChange[splitChange["Model"] == "Visual"]
Visual_Likelihoods = Visual_Likelihoods["Likelihood"]




from scipy.stats import tukey_hsd
res = tukey_hsd(Utility_100_Likelihoods, Visual_Likelihoods)
print(res)

#print(splitChange["Likelihood"].min())
#print(splitChange["Likelihood"].max())
splitChange = splitChange[splitChange["Likelihood"] < 1.1]
Utility_80_Likelihoods = splitChange[splitChange["Model"] == "Utility 80"]
Utility_80_Likelihoods = Utility_80_Likelihoods["Likelihood"]
res = tukey_hsd(Utility_100_Likelihoods, Utility_80_Likelihoods)
print(res)

Utility_60_Likelihoods = splitChange[splitChange["Model"] == "Utility 60"]
Utility_60_Likelihoods = Utility_60_Likelihoods["Likelihood"]
res = tukey_hsd(Utility_100_Likelihoods, Utility_60_Likelihoods)
print(res)

Utility_40_Likelihoods = splitChange[splitChange["Model"] == "Utility 40"]
Utility_40_Likelihoods = Utility_40_Likelihoods["Likelihood"]
res = tukey_hsd(Utility_100_Likelihoods, Utility_40_Likelihoods)
print(res)


palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, n_colors=4) 
sns.barplot(data=splitChange, x="Model", y="Likelihood", errorbar=('ci', 90), order=order, hue="Split", palette=palette, ax=axes[2])
axes[2].set_ylim(0.7, 0.8)

axes[0].set_title("Detection Probability by Utility Difference", fontsize=14)
axes[1].set_title("Detection Probability by Visual Difference", fontsize=14)
axes[2].set_title("Likelihood by Model and Data Split", fontsize=14)

axes[0].set_ylabel("Detection Probability", fontsize=12)
axes[1].set_ylabel("Detection Probability", fontsize=12)
axes[2].set_ylabel("Log Likelihood", fontsize=12)

axes[0].set_xlabel("Utility Difference", fontsize=12)
axes[1].set_xlabel("Visual Difference", fontsize=12)
axes[2].set_xlabel("Log Likelihood", fontsize=12)

plt.show()