import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

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
        num_util_observations += 1

        if(trial['Block'] != old_block):
            old_block = trial['Block'] 
            num_util_observations = 0

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

            d = pd.DataFrame([[id, num_util_observations, chosen_ev - unchosen_ev, chosen_var - unchosen_var,  1]], columns=selectionProbabilityColumns)
            selectionProbability = pd.concat([selectionProbability, d], ignore_index=True)
            d = pd.DataFrame([[id, num_util_observations, unchosen_ev - chosen_ev, unchosen_var - chosen_var,  0]], columns=selectionProbabilityColumns)
            selectionProbability = pd.concat([selectionProbability, d], ignore_index=True)


from scipy.stats import pearsonr 

correlationColumns = ["Number of Utility Observations", "Pearson Correlation"]
correlations = pd.DataFrame([], correlationColumns)

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

from scipy.optimize import curve_fit
correlations = correlations.dropna()
correlations = correlations.reset_index()
print(correlations)

# Fitting
model = lambda x, A, x0, offset:  offset+A*np.log(x-x0)
popt, pcov = curve_fit(model, correlations["Number of Utility Observations"].values, 
                              correlations["Pearson Correlation"].values, p0=[1,0,2])
#plot fit
x = np.linspace(correlations["Number of Utility Observations"].values.min(), correlations["Number of Utility Observations"].values.max(),250)
plt.plot(x,model(x,*popt), label="Regression")
sns.scatterplot(correlations, x="Number of Utility Observations", y="Pearson Correlation")

#sns.lmplot(x="Number of Utility Observations", y="Pearson Correlation", data=correlations, order=2, ci=None, scatter_kws={"s": 80})
plt.ylabel("Correlation of Variance and Selection", fontsize=12)
plt.xlabel("Number of Utility Observations", fontsize=12)

plt.title("Correlation by Number of Utility Observations", fontsize=14)
plt.show()