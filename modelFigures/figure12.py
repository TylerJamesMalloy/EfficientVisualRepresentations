import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
#sns.set()

fig, axes = plt.subplots(nrows=1, ncols=2)



modelLearned = pd.read_pickle("./modelLearned.pkl")
modelLearned["Log Likelihood"] = -1 * modelLearned["Log Likelihood"] / 3
palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, n_colors=4) 

modelBehaviorAversion = pd.read_pickle("./modelBehaviorChange.pkl")

sns.barplot(modelLearned, x="Model", y="Log Likelihood", hue="Split", palette=palette, ax=axes[1])

axes[1].set_ylabel("Log Likelihood", fontsize=12)
axes[1].set_xlabel("Data Split", fontsize=12)
axes[1].set_title("Log Likelihood by Data Split", fontsize=14)

correlationColumns = ["Number of Utility Observations", "Pearson Correlation"]
SeekingCorrelations = pd.DataFrame([
                             [2, 0.2],
                             [3, 0.24],
                             [4, 0.26],
                             [5, 0.30],
                             [6, 0.32],
                             [7, 0.31],
                             [8, 0.44],
                             [9, 0.47],
                             ], columns=correlationColumns)

AverseCorrelations = pd.DataFrame([
                             [2, -0.2],
                             [3, -0.24],
                             [4, -0.26],
                             [5, -0.30],
                             [6, -0.32],
                             [7, -0.31],
                             [8, -0.44],
                             [9, -0.47],
                             ], columns=correlationColumns)

from scipy.optimize import curve_fit

model = lambda x, A, offset:  offset+A*x

popt, pcov = curve_fit(model, AverseCorrelations["Number of Utility Observations"].values, 
                              AverseCorrelations["Pearson Correlation"].values, p0=[1,2])
#plot fit
x = np.linspace(AverseCorrelations["Number of Utility Observations"].values.min(), AverseCorrelations["Number of Utility Observations"].values.max(),250)
axes[0].plot(x, model(x,*popt), label="B=4", color="blue")
axes[0].scatter(AverseCorrelations["Number of Utility Observations"].tolist(), AverseCorrelations["Pearson Correlation"].tolist())

# Fitting

popt, pcov = curve_fit(model, SeekingCorrelations["Number of Utility Observations"].values, 
                              SeekingCorrelations["Pearson Correlation"].values, p0=[1,2])
#plot fit
x = np.linspace(SeekingCorrelations["Number of Utility Observations"].values.min(), SeekingCorrelations["Number of Utility Observations"].values.max(),250)
axes[0].plot(x, model(x,*popt), label="B=100", color="orange")
axes[0].scatter(SeekingCorrelations["Number of Utility Observations"].tolist(), SeekingCorrelations["Pearson Correlation"].tolist())

#sns.lmplot(x="Number of Utility Observations", y="Pearson Correlation", data=correlations, order=2, ci=None, scatter_kws={"s": 80})
axes[0].set_ylabel("Correlation of Variance and Selection", fontsize=12)
axes[0].set_xlabel("Number of Utility Observations", fontsize=12)
axes[0].set_title("Correlation by Number of Utility Observations", fontsize=14)

axes[0].legend()

plt.show()


import scipy.stats as stats
# ANOVA
result = stats.f_oneway(modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE"],
                        modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE 80"],
                        modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE 60"],
                        modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE 40"])
print(result)

res = stats.tukey_hsd(modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE"],
                        modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE 80"],
                        modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE 60"],
                        modelLearned['Log Likelihood'][modelLearned['Model'] == "UBVAE 40"])
print(res)

assert(False)
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
#sns.set()
import scipy.stats as stats 

fig, axes = plt.subplots(nrows=1, ncols=2)


modelLearned = pd.read_pickle("./modelLearned.pkl")
modelLearned = modelLearned.reset_index()
modelLearned["Log Likelihood"] = -1 * modelLearned["Log Likelihood"] 
modelLearned = modelLearned.drop(columns=["level_0"])
modelLearned = modelLearned.drop(columns=["index"])


palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, n_colors=4) 
print(palette)


sns.barplot(data=modelLearned, x="Model", y="Log Likelihood", hue='Split', pallette=palette)
plt.title("Learned")
plt.show()


