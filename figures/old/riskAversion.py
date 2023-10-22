import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

df = pd.read_csv("../data/participantData.csv")

df = df[df['Experiment Type'] == 'Decision-Making']

# 145 Decision Making Participants 

ids = df["Id"].unique()

#riskAversionColumns = ["Chosen EV", "Chosen Var", "Unchosen EV", "Unchosen Var"]
riskAversionColumns = ["Outcome Expected Value (Chosen - Unchosen)", "Outcome Variance (Chosen - Unchosen)"]
riskAversion = pd.DataFrame([], columns=riskAversionColumns)
riskAversionUnrounded = pd.DataFrame([], columns=riskAversionColumns)

selectionProbabilityColumns = ["Utility Difference", "Chosen"]
selectionProbability = pd.DataFrame([], columns=selectionProbabilityColumns)

cptEffectColumns = ["Utility Difference - Weighted Variance", "Chosen"]
cptEffect = pd.DataFrame([], columns=cptEffectColumns)

for id in ids:
    idf = df[df["Id"] == id]
    idf = idf.tail(180) # Skip learning trials 
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

            d = pd.DataFrame([[chosen_ev - unchosen_ev, 1]], columns=selectionProbabilityColumns)
            selectionProbability = pd.concat([selectionProbability, d], ignore_index=True)
            d = pd.DataFrame([[unchosen_ev - chosen_ev, 0]], columns=selectionProbabilityColumns)
            selectionProbability = pd.concat([selectionProbability, d], ignore_index=True)
            
            chosen = chosen_ev - 0.2 * chosen_var
            unchosen = unchosen_ev - 0.2 * unchosen_var

            d = pd.DataFrame([[chosen - unchosen, 1]], columns=cptEffectColumns)
            cptEffect = pd.concat([cptEffect, d], ignore_index=True)
            d = pd.DataFrame([[unchosen - chosen, 0]], columns=cptEffectColumns)
            cptEffect = pd.concat([cptEffect, d], ignore_index=True)

            ev_diff = chosen_ev - unchosen_ev
            var_diff = chosen_var - unchosen_var

            d = pd.DataFrame([[ev_diff, var_diff]], columns=riskAversionColumns)
            riskAversionUnrounded = pd.concat([riskAversionUnrounded, d], ignore_index=True)

            ev_diff = round(ev_diff, 1)
            var_diff = round(var_diff, 1)

            d = pd.DataFrame([[ev_diff, var_diff]], columns=riskAversionColumns)
            riskAversion = pd.concat([riskAversion, d], ignore_index=True)


fig, ax = plt.subplots()
sns.regplot(data=riskAversionUnrounded, x="Outcome Variance (Chosen - Unchosen)", y="Outcome Expected Value (Chosen - Unchosen)", scatter=False, ci=99)
#sns.scatterplot(data=riskAversionUnrounded, x="Outcome Variance (Chosen - Unchosen)", y="Outcome Expected Value (Chosen - Unchosen)", alpha=0.01)

group = riskAversion.groupby("Outcome Variance (Chosen - Unchosen)")["Outcome Expected Value (Chosen - Unchosen)"].mean()
xd = group.index.values.tolist()
yd = group.values.tolist()
yerrd = riskAversion.groupby("Outcome Variance (Chosen - Unchosen)")["Outcome Expected Value (Chosen - Unchosen)"].var().values.tolist()

ax.errorbar(xd, yd, yerr=yerrd, fmt='none', capsize=5, zorder=1, color='C0')
ax.scatter(xd, yd, color='C0')

ax.set_title("Participant Chosen Option Expected Value by Variance", fontsize=18)
ax.set_xlabel("Outcome Variance (Chosen - Unchosen)", fontsize=16)
ax.set_ylabel("Outcome Expected Value \n (Chosen - Unchosen)", fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.show()

selectionProbability = selectionProbability[selectionProbability["Utility Difference"] > -0.8]
selectionProbability = selectionProbability[selectionProbability["Utility Difference"] < 0.8]

fig, ax = plt.subplots()
group = selectionProbability.groupby("Utility Difference")["Chosen"].mean()
xd = group.index.values.tolist()
yd = group.values.tolist()
ax.scatter(xd, yd, color='C0')
plt.show()
