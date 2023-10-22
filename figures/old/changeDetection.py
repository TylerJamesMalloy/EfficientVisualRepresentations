import os 

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import scipy.stats as st

dataFrame = pd.read_csv("./clean/participantData.csv")
goodParticipants = pd.read_csv("./clean/goodParticipants.csv").to_numpy()[:,1]

dataFrame["Left Stimulus Marbles"] = dataFrame["Left Stimulus Marbles"].apply(eval)
dataFrame["Right Stimulus Marbles"] = dataFrame["Right Stimulus Marbles"].apply(eval)
dataFrame["New Stimulus Marbles"] = dataFrame["New Stimulus Marbles"].apply(eval)

dataFrame = dataFrame.loc[dataFrame["Id"].isin(goodParticipants)]

change_data = dataFrame[dataFrame["Experiment Type"] == 'Decision-Making']
change_data = change_data[change_data["Trial Type"] == 'Change Detection']
change_data = change_data[change_data["Changed"] == 1]

changeUtilityMeanDifference = []

for _, data in change_data.iterrows():
    if(data['Changed Index'] == 0):
        originalMean = np.mean(data["Left Stimulus Marbles"])
    else:
        originalMean = np.mean(data["Right Stimulus Marbles"])

    changeUtilityMeanDifference.append((np.mean(data["New Stimulus Marbles"]) - originalMean) ** 2)


change_data["Changed Utility Mean Squared Difference"] = changeUtilityMeanDifference

change_data["Probability of Detecting Change"] = change_data["Key Pressed"] == "k"
change_data["Probability of Detecting Change"] = change_data["Probability of Detecting Change"].astype(float)

means = change_data.groupby("Changed Utility Mean Squared Difference").mean()

sns.scatterplot(means, x="Changed Utility Mean Squared Difference", y="Probability of Detecting Change")
sns.regplot(data=change_data, x="Changed Utility Mean Squared Difference", y="Probability of Detecting Change", scatter=False)
plt.title("Probability of Detecting Change by\n Changed Utility Mean Squared Difference")

plt.show()

print(st.pearsonr(x=change_data["Probability of Detecting Change"], y=change_data["Changed Utility Mean Squared Difference"]))


