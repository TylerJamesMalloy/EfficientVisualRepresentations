import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
#sns.set()

fig, axes = plt.subplots(nrows=1, ncols=2)



modelLearned = pd.read_pickle("./modelLearned.pkl")
modelLearned["Log Likelihood"] = -1 * modelLearned["Log Likelihood"]
palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, n_colors=4) 

modelBehaviorAversion = pd.read_pickle("./modelBehaviorChange.pkl")

sns.barplot(modelLearned, x="Model", y="Log Likelihood", hue="Split", palette=palette, ax=axes[1])

axes[1].set_ylabel("Log Likelihood", fontsize=12)
axes[1].set_xlabel("Data Split", fontsize=12)
axes[1].set_title("Log Likelihood by Data Split", fontsize=14)

correlationColumns = ["Number of Utility Observations", "Pearson Correlation"]
correlations = pd.DataFrame([[1, 0.05],
                             [2, 0.08],
                             [3, 0.14],
                             [4, 0.16],
                             [5, 0.2],
                             [6, 0.2],
                             [7, 0.21],
                             [8, 0.21],
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


