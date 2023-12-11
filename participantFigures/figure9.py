import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import scipy.stats as stats

df = pd.read_csv("../data/participantData.csv")
"""'
Unnamed: 0', 'Id', 'Experiment Type', 'Marble Set', 'Block', 'Trial',
       'Trial Type', 'Reaction Time', 'Left Stimulus', 'Right Stimulus',
       'Left Stimulus Marbles', 'Right Stimulus Marbles', 'New Stimulus',
       'New Stimulus Marbles', 'Changed', 'Changed Index', 'Key Pressed',
       'Reward', 'Correct'
"""
df = df[df['Experiment Type'] == 'Learning']

# 145 Decision Making Participants 



#order = ["EUT", "Utility 40", "Utility 60", "Utility 80", "Utility"]

fitEquivalence = pd.read_pickle("./fitEquivalence.pkl")
fitEquivalence = fitEquivalence.reset_index()
fitEquivalence.loc[fitEquivalence['Split'] == 80, 'Model'] = "Utility 40"
fitEquivalence.loc[fitEquivalence['Split'] == 60, 'Model'] = "Utility 60"
fitEquivalence.loc[fitEquivalence['Split'] == 40, 'Model'] = "Utility 80"

fitEquivalence.loc[fitEquivalence['Model'] == "Utility 40", 'Likelihood'] = np.log(fitEquivalence.loc[fitEquivalence['Model'] == "Utility 40", 'Likelihood'])
fitEquivalence.loc[fitEquivalence['Model'] == "Utility 60", 'Likelihood'] = np.log(fitEquivalence.loc[fitEquivalence['Model'] == "Utility 60", 'Likelihood'])
fitEquivalence.loc[fitEquivalence['Model'] == "Utility 80", 'Likelihood'] = np.log(fitEquivalence.loc[fitEquivalence['Model'] == "Utility 80", 'Likelihood'])
fitEquivalence.loc[fitEquivalence['Model'] == "Utility", 'Likelihood'] = np.log(fitEquivalence.loc[fitEquivalence['Model'] == "Utility", 'Likelihood'])

#fitEquivalence["Likelihood"] = np.log(fitEquivalence["Likelihood"])

order = ["Visual", "Utility 40", "Utility 60", "Utility 80", "Utility"]
palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, n_colors=4) 
fig, axes = plt.subplots(nrows=1, ncols=2)
sns.barplot(fitEquivalence, x="Model", y="Likelihood", hue="Split", order=order, palette=palette, ax=axes[1])
axes[1].set_ylabel("Log Likelihood", fontsize=12)
axes[1].set_xlabel("Model and Data Split", fontsize=12)
axes[1].set_title("Log Likelihood by Model and Data Split", fontsize=14)

fitEquivalence.replace([np.inf, -np.inf], np.nan, inplace=True)
fitEquivalence.dropna(inplace=True)

# ANOVA
result = stats.f_oneway(fitEquivalence['Likelihood'][fitEquivalence['Model'] == "Utility"],
                        fitEquivalence['Likelihood'][fitEquivalence['Model'] == "Utility 80"],
                        fitEquivalence['Likelihood'][fitEquivalence['Model'] == "Utility 60"],
                        fitEquivalence['Likelihood'][fitEquivalence['Model'] == "Utility 40"],
                        fitEquivalence['Likelihood'][fitEquivalence['Model'] == "Visual"])
print(result)

res = stats.tukey_hsd(fitEquivalence['Likelihood'][fitEquivalence['Model'] == "Utility"],
                        fitEquivalence['Likelihood'][fitEquivalence['Model'] == "Utility 80"],
                        fitEquivalence['Likelihood'][fitEquivalence['Model'] == "Utility 60"],
                        fitEquivalence['Likelihood'][fitEquivalence['Model'] == "Utility 40"],
                        fitEquivalence['Likelihood'][fitEquivalence['Model'] == "Visual"])
print(res)

#fitEquivalence.to_pickle("../stats/participantEquivalence.pkl")

"""
PearsonRResult(statistic=0.3596496949735051, pvalue=0.18795346210607378)
PearsonRResult(statistic=0.39340762378206395, pvalue=0.14684965504567643)
PearsonRResult(statistic=0.3942308423479115, pvalue=0.14593019959391948)
PearsonRResult(statistic=0.4684338383199314, pvalue=0.07821278021655274)
PearsonRResult(statistic=0.7464973200763091, pvalue=0.002162801066850038)
PearsonRResult(statistic=0.6210363601304636, pvalue=0.01777227382959907)
"""
correlationColumns = ["Number of Utility Observations", "Pearson Correlation"]
correlations = pd.DataFrame([[2, 0.3],
                             [3, 0.359],
                             [4, 0.393],
                             [5, 0.394],
                             [6, 0.746],
                             [7, 0.621],
                             [8, 0.821],
                             ], columns=correlationColumns)

from scipy.optimize import curve_fit

# Fitting
model = lambda x, A, x0, offset:  offset+A*(x-x0)
popt, pcov = curve_fit(model, correlations["Number of Utility Observations"].values, 
                              correlations["Pearson Correlation"].values, p0=[1,0,2])
#plot fit
x = np.linspace(correlations["Number of Utility Observations"].values.min(), correlations["Number of Utility Observations"].values.max(),250)
axes[0].plot(x, model(x,*popt), label="Regression")
axes[0].scatter(correlations["Number of Utility Observations"].tolist(), correlations["Pearson Correlation"].tolist())

#sns.lmplot(x="Number of Utility Observations", y="Pearson Correlation", data=correlations, order=2, ci=None, scatter_kws={"s": 80})
axes[0].set_ylabel("Correlation of Utility and Detection", fontsize=12)
axes[0].set_xlabel("Number of Utility Observations", fontsize=12)
axes[0].set_title("Correlation by Number of Utility Observations", fontsize=14)

plt.show()


"""ids = df["Id"].unique()
print(len(ids))

#riskAversionColumns = ["Chosen EV", "Chosen Var", "Unchosen EV", "Unchosen Var"]
changeDetectionColumns = ["Id", "Utility Observations", "Visual Difference", "Utility Difference", "Detected"]
changeDetection = pd.DataFrame([], columns=changeDetectionColumns)

for id in ids:
    idf = df[df["Id"] == id]
    idf = idf.tail(180) # Skip learning trials 
    old_block = 1.0 
    num_util_observations = 0
    for _, trial in idf.iterrows(): 
        if(trial['Trial Type'] == 'Utility Selection'):
            num_util_observations += 1

        if(trial['Block'] != old_block):
            old_block = trial['Block'] 
            num_util_observations = 0

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

            d = pd.DataFrame([[id, num_util_observations, vis_diff, ev_diff, correct]], columns=changeDetectionColumns)
            changeDetection = pd.concat([changeDetection, d])  

group = changeDetection.groupby(["Id"]).mean()['Detected']
#good = group[group > 0.7]
#good = good.keys().tolist()
#changeDetection = changeDetection[changeDetection["Id"].isin(good)]

correlationColumns = ["Number of Utility Observations", "Pearson Correlation"]
correlations = pd.DataFrame([], correlationColumns)


for observations in range(3,10):
    data = changeDetection[changeDetection["Utility Observations"] == observations]
    grouped = data.groupby(['Utility Difference']).mean()["Detected"]

    regDataColumns = ["Utility Difference", "Detection Probability"]
    regData = pd.DataFrame([], columns=regDataColumns)
    for key, value in grouped.items():
        d = pd.DataFrame([[key, value]], columns=regDataColumns) 
        regData = pd.concat([regData, d], ignore_index=True)

    r = pearsonr(regData["Utility Difference"], regData["Detection Probability"])

    
    d = pd.DataFrame([[observations, r.statistic]], columns=correlationColumns)
    correlations = pd.concat([d, correlations])

sns.regplot(data=correlations, x="Number of Utility Observations", y="Pearson Correlation")
plt.show()"""
