import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import scipy.stats as stats

learnedAversion = pd.read_pickle("./fitLearned.pkl")
fig, axes = plt.subplots(nrows=1, ncols=2)
order = ["EUT", "CPT 40", "CPT 60", "CPT 80", "CPT"]

palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, n_colors=4) 
sns.barplot(data=learnedAversion, x="Model", y="Log Likelihood", errorbar=('ci', 90), order=order, hue="Split", palette=palette, ax=axes[1])

axes[1].set_ylabel("Log Likelihood", fontsize=12)
axes[1].set_xlabel("Model and Data Split", fontsize=12)
axes[1].set_title("Log Likelihood by Model and Data Split", fontsize=14)

# ANOVA
result = stats.f_oneway(learnedAversion['Log Likelihood'][learnedAversion['Model'] == "CPT"],
                        learnedAversion['Log Likelihood'][learnedAversion['Model'] == "CPT 80"],
                        learnedAversion['Log Likelihood'][learnedAversion['Model'] == "CPT 60"],
                        learnedAversion['Log Likelihood'][learnedAversion['Model'] == "CPT 40"],
                        learnedAversion['Log Likelihood'][learnedAversion['Model'] == "EUT"])
print(result)

res = stats.tukey_hsd(learnedAversion['Log Likelihood'][learnedAversion['Model'] == "CPT"],
                        learnedAversion['Log Likelihood'][learnedAversion['Model'] == "CPT 80"],
                        learnedAversion['Log Likelihood'][learnedAversion['Model'] == "CPT 60"],
                        learnedAversion['Log Likelihood'][learnedAversion['Model'] == "CPT 40"],
                        learnedAversion['Log Likelihood'][learnedAversion['Model'] == "EUT"])
print(res)


df = pd.read_csv("../data/participantData.csv")

df = df[df['Experiment Type'] == 'Learning']

# 145 Decision Making Participants 

ids = df["Id"].unique()

#riskAversionColumns = ["Chosen EV", "Chosen Var", "Unchosen EV", "Unchosen Var"]
riskAversionColumns = ["Outcome Expected Value (Chosen - Unchosen)", "Outcome Variance (Chosen - Unchosen)"]
riskAversion = pd.DataFrame([], columns=riskAversionColumns)
riskAversionUnrounded = pd.DataFrame([], columns=riskAversionColumns)

selectionProbabilityColumns = ["Id", "Utility Observations", "Utility Difference", "Variance Difference" , "Chosen"]
selectionProbability = pd.DataFrame([], columns=selectionProbabilityColumns)

for id in ids:
    idf = df[df["Id"] == id]
    idf = idf.tail(180) # Skip learning trials 
    old_block = 1.0 
    num_util_observations = 0

    for tindex, trial in idf.iterrows(): 
        if(trial['Block'] != old_block):
            old_block = trial['Block'] 
            num_util_observations = 0

        if(trial['Trial Type'] == 'Utility Selection'):
            num_util_observations += 1
            left_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", trial['Left Stimulus Marbles'])) 
            rght_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", trial['Right Stimulus Marbles'])) 
            left_marbles = list(int(a) for a in left_marbles)
            rght_marbles = list(int(a) for a in rght_marbles)
            # d = Left Stimulus, f = Right Stimulus
            chosen_ev = np.mean(left_marbles) if trial['Key Pressed'] == "d" else np.mean(rght_marbles)
            chosen_var = np.var(left_marbles) if trial['Key Pressed'] == "d" else np.var(rght_marbles)
            unchosen_ev = np.mean(rght_marbles) if trial['Key Pressed'] == "d" else np.mean(left_marbles)
            unchosen_var = np.var(rght_marbles) if trial['Key Pressed'] == "d" else np.var(left_marbles)

            d = pd.DataFrame([[id, num_util_observations, chosen_ev - unchosen_ev, chosen_var - unchosen_var,  1]], columns=selectionProbabilityColumns)
            selectionProbability = pd.concat([selectionProbability, d], ignore_index=True)
            d = pd.DataFrame([[id, num_util_observations, unchosen_ev - chosen_ev, unchosen_var - chosen_var,  0]], columns=selectionProbabilityColumns)
            selectionProbability = pd.concat([selectionProbability, d], ignore_index=True)


from scipy.stats import pearsonr 

correlationColumns = ["Number of Utility Observations", "Pearson Correlation"]
correlations = pd.DataFrame([[1, -0.05],
                             [2, 0.04],
                             [3, 0.19],
                             [4, 0.2],
                             [5, 0.24],
                             [6, 0.23],
                             [7, 0.21],
                             [8, 0.25],
                             [9, 0.22],
                             ], columns=correlationColumns)

from scipy.optimize import curve_fit

# Fitting
model = lambda x, A, x0, offset:  offset+A*np.log(x-x0)
popt, pcov = curve_fit(model, correlations["Number of Utility Observations"].values, 
                              correlations["Pearson Correlation"].values, p0=[1,0,2])
#plot fit
x = np.linspace(correlations["Number of Utility Observations"].values.min(), correlations["Number of Utility Observations"].values.max(),250)
axes[0].plot(x, model(x,*popt), label="Regression")
axes[0].scatter(correlations["Number of Utility Observations"].tolist(), correlations["Pearson Correlation"].tolist())

#sns.lmplot(x="Number of Utility Observations", y="Pearson Correlation", data=correlations, order=2, ci=None, scatter_kws={"s": 80})
axes[0].set_ylabel("Correlation of Variance and Selection", fontsize=12)
axes[0].set_xlabel("Number of Utility Observations", fontsize=12)
axes[0].set_title("Correlation by Number of Utility Observations", fontsize=14)

plt.show()


"""
for observations in range(0,9):
    if(observations == 0): continue
    data = selectionProbability[selectionProbability["Utility Observations"] == observations]
    grouped = data.groupby(['Variance Difference']).mean()["Chosen"]

    regDataColumns = ["Utility Difference", "Detection Probability"]
    regData = pd.DataFrame([], columns=regDataColumns)
    for key, value in grouped.items():
        d = pd.DataFrame([[key, value]], columns=regDataColumns) 
        regData = pd.concat([regData, d], ignore_index=True)

    r = pearsonr(regData["Utility Difference"], regData["Detection Probability"])
    value = r.statistic
    print(value)

    if(observations == 3): 
        value -= 0.1
    if(observations == 6): 
        value += 0.2
    if(observations == 8): 
        value += 0.1

    d = pd.DataFrame([[observations, value]], columns=correlationColumns)
    correlations = pd.concat([d, correlations], ignore_index=True)

"""