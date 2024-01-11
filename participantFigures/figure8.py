import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import scipy.stats as stats
from scipy.stats import pearsonr 

learnedAversion = pd.read_pickle("./fitLearned.pkl")
fig, axes = plt.subplots(nrows=1, ncols=2)
order = ["EUT", "CPT 40", "CPT 60", "CPT 80", "CPT"]

learnedAversion["Log Likelihood"] = learnedAversion["Log Likelihood"] / 3
learnedAversion.to_pickle("./fitLearned.pkl")

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
#print(result)

res = stats.tukey_hsd(learnedAversion['Log Likelihood'][learnedAversion['Model'] == "CPT"],
                        learnedAversion['Log Likelihood'][learnedAversion['Model'] == "CPT 80"],
                        learnedAversion['Log Likelihood'][learnedAversion['Model'] == "CPT 60"],
                        learnedAversion['Log Likelihood'][learnedAversion['Model'] == "CPT 40"],
                        learnedAversion['Log Likelihood'][learnedAversion['Model'] == "EUT"])
#print(res)

learnedAversion.to_pickle("../stats/participantLearned.pkl")


df = pd.read_csv("../data/participantData.csv")

correlationColumns = ["Number of Utility Observations", "Pearson Correlation"]

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


correlationColumns = ["Number of Utility Observations", "Pearson Correlation"]
correlations = pd.DataFrame([], columns=correlationColumns)

riskSeeking = learnedAversion[learnedAversion["Aversion Coefficient"] < 0]
riskAverse = learnedAversion[learnedAversion["Aversion Coefficient"] > 0]

riskSeekingCorrelations = [[] for _ in range(11)]
riskAverseCorrelations = [[] for _ in range(11)]

for idx, participants in enumerate([riskAverse, riskSeeking]):
    ids = participants["Id"].unique()
    for observations in range(0,10):
        data = selectionProbability[selectionProbability["Id"].isin(ids)]
        data = data[data["Utility Observations"] == observations]
        grouped = data.groupby(['Variance Difference']).mean()["Chosen"]

        regDataColumns = ["Utility Difference", "Detection Probability"]
        regData = pd.DataFrame([], columns=regDataColumns)
        for key, value in grouped.items():
            d = pd.DataFrame([[key, value]], columns=regDataColumns) 
            regData = pd.concat([regData, d], ignore_index=True)

        if(len(regData["Detection Probability"]) < 3): continue 
        r = pearsonr(regData["Utility Difference"], regData["Detection Probability"])
        value = r.statistic
        pvalue = r.pvalue 
        if(pvalue < 0.05):
            if(idx == 0):
                riskAverseCorrelations[observations].append(value)
            else:
                riskSeekingCorrelations[observations].append(value)

#print(riskAverseCorrelations)
#[[], [0.1946640823989317], [], [0.19760989075150953], [0.2021442830453205], [], [], [0.3766844154512214], [], [0.3151838452818285]]

#print(riskSeekingCorrelations)
#[[], [-0.26688728611311785], [-0.2826203310360188], [], [], [-0.2466797240706582], [-0.3504848145347185], [], [-0.45531854000426264], [-0.49242773417365754]]

correlationColumns = ["Number of Utility Observations", "Pearson Correlation"]
SeekingCorrelations = pd.DataFrame([
                             [1, 0.1946640823989317],
                             [3, 0.19760989075150953],
                             [4, 0.2021442830453205],
                             [7, 0.3766844154512214],
                             [9, 0.3151838452818285]
                             ], columns=correlationColumns)

AverseCorrelations = pd.DataFrame([
                             [1, -0.26688728611311785],
                             [2, -0.2826203310360188],
                             [5, -0.2466797240706582],
                             [6, -0.3504848145347185],
                             [8, -0.45531854000426264],
                             [9, -0.49242773417365754],
                             ], columns=correlationColumns)

from scipy.optimize import curve_fit

# Fitting
model = lambda x, A, offset:  offset+A*x
popt, pcov = curve_fit(model, AverseCorrelations["Number of Utility Observations"].values, 
                              AverseCorrelations["Pearson Correlation"].values, p0=[1,2])
#plot fit
x = np.linspace(AverseCorrelations["Number of Utility Observations"].values.min(), AverseCorrelations["Number of Utility Observations"].values.max(),250)
axes[0].plot(x, model(x,*popt), label="Risk Averse", color="blue")
axes[0].scatter(AverseCorrelations["Number of Utility Observations"].tolist(), AverseCorrelations["Pearson Correlation"].tolist())

popt, pcov = curve_fit(model, SeekingCorrelations["Number of Utility Observations"].values, 
                              SeekingCorrelations["Pearson Correlation"].values, p0=[1,2])
#plot fit
x = np.linspace(SeekingCorrelations["Number of Utility Observations"].values.min(), SeekingCorrelations["Number of Utility Observations"].values.max(),250)
axes[0].plot(x, model(x,*popt), label="Risk Seeking", color="orange")
axes[0].scatter(SeekingCorrelations["Number of Utility Observations"].tolist(), SeekingCorrelations["Pearson Correlation"].tolist())

#sns.lmplot(x="Number of Utility Observations", y="Pearson Correlation", data=correlations, order=2, ci=None, scatter_kws={"s": 80})
axes[0].set_ylabel("Correlation of Variance and Selection", fontsize=12)
axes[0].set_xlabel("Number of Utility Observations", fontsize=12)
axes[0].set_title("Correlation by Number of Utility Observations", fontsize=14)
axes[0].legend()

plt.show()





