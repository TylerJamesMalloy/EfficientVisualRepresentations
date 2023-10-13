import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 


fig, axes = plt.subplots(nrows=1, ncols=2)

"""participantAversion = pd.read_pickle(".\participantAversion_e1.pkl")
participantAversion = participantAversion.reset_index()
sns.histplot(participantAversion, x="Risk Aversion Coefficient", ax=axes[1])
plt.show()

assert(False)"""
axes[1].set_ylabel("Participant Count")
axes[1].set_xlabel("Risk Aversion Coefficient")
axes[1].set_title("Participant Risk Aversion Coefficients")

df = pd.read_csv("../clean/participantData.csv")
"""'
Unnamed: 0', 'Id', 'Experiment Type', 'Marble Set', 'Block', 'Trial',
       'Trial Type', 'Reaction Time', 'Left Stimulus', 'Right Stimulus',
       'Left Stimulus Marbles', 'Right Stimulus Marbles', 'New Stimulus',
       'New Stimulus Marbles', 'Changed', 'Changed Index', 'Key Pressed',
       'Reward', 'Correct'
"""
#df = df[df['Experiment Type'] == 'Decision-Making']

# 145 Decision Making Participants 

ids = df["Id"].unique()

#riskAversionColumns = ["Chosen EV", "Chosen Var", "Unchosen EV", "Unchosen Var"]
riskAversionColumns = ["Outcome Expected Value (Chosen - Unchosen)", "Outcome Variance (Chosen - Unchosen)"]
riskAversion = pd.DataFrame([], columns=riskAversionColumns)
riskAversionUnrounded = pd.DataFrame([], columns=riskAversionColumns)

selectionProbabilityColumns = ["Utility Difference", "Chosen", "Type"]
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

            d = pd.DataFrame([[chosen_ev - unchosen_ev, 1, "Unweighted"]], columns=selectionProbabilityColumns)
            selectionProbability = pd.concat([selectionProbability, d], ignore_index=True)
            d = pd.DataFrame([[unchosen_ev - chosen_ev, 0, "Unweighted"]], columns=selectionProbabilityColumns)
            selectionProbability = pd.concat([selectionProbability, d], ignore_index=True)
            
            
unweighted = selectionProbability[selectionProbability["Type"] == "Unweighted"]
unweighted = unweighted[unweighted["Utility Difference"] > -0.8]
unweighted = unweighted[unweighted["Utility Difference"] < 0.8]

fig, ax = plt.subplots()
group = unweighted.groupby("Utility Difference")["Chosen"].mean()
xd = group.index.values.tolist()
yd = group.values.tolist()

#yerrd = unweighted.groupby("Utility Difference")["Chosen"].std().values.tolist()
#axes[0].errorbar(xd, yd, yerr=yerrd, fmt='none', capsize=5, zorder=1, color='C0')

axes[0].scatter(xd, yd, color='C0')
sns.regplot(x=xd, y=yd, order=3, ax=axes[0], label="Unweighted")

axes[0].set_ylabel("Participant Choice Probability")
axes[0].set_xlabel("Utility Difference (Chosen - Unchosen)")
axes[0].set_title("Participant Choice Probability by Utility Difference")

plt.show()


