import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
#sns.set()

fig, axes = plt.subplots(nrows=1, ncols=2)

"""
PearsonRResult(statistic=0.24120487340427138, pvalue=0.35100859787905775)
PearsonRResult(statistic=0.45602510831180043, pvalue=0.0657993447615786)
PearsonRResult(statistic=0.5272146744659986, pvalue=0.029651396822118935)
PearsonRResult(statistic=0.4987614697810883, pvalue=0.041552249698749426)
PearsonRResult(statistic=0.5724094417642499, pvalue=0.016337713754722632)
PearsonRResult(statistic=0.6146594291408876, pvalue=0.008649954682762298)
"""
# C:\Users\Tyler\Desktop\Projects\EfficientVisualRepresentations\modelFigures\modelEquivalence.pkl
modelEquivalence = pd.read_pickle("./modelEquivalence.pkl")
modelEquivalence["Log Likelihood"] = -1 * modelEquivalence["Log Likelihood"]
palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, n_colors=4) 

modelBehaviorAversion = pd.read_pickle("./modelBehaviorChange.pkl")

sns.barplot(modelEquivalence, x="Model", y="Log Likelihood", hue="Split", palette=palette, ax=axes[1])

axes[1].set_ylabel("Log Likelihood", fontsize=12)
axes[1].set_xlabel("Data Split", fontsize=12)
axes[1].set_title("Log Likelihood by Data Split", fontsize=14)

correlationColumns = ["Number of Utility Observations", "Variance Bias Correlation"]
correlations = pd.DataFrame([[1, 0.34],
                             [2, 0.45],
                             [3, 0.52],
                             [4, 0.49],
                             [5, 0.55],
                             [6, 0.48],
                             [7, 0.55],
                             [8, 0.57],
                             [9, 0.61],
                             
                             ], columns=correlationColumns)

sns.regplot(correlations, x="Number of Utility Observations", y="Variance Bias Correlation", scatter=False, ax=axes[0])
sns.scatterplot(correlations, x="Number of Utility Observations", y="Variance Bias Correlation", ax=axes[0])

axes[0].set_ylabel("Utility Bias in Change Detection", fontsize=12)
axes[0].set_xlabel("Number of Utility Observations", fontsize=12)
axes[0].set_title("Utility Bias in Change Detection by Number of Utility Observations", fontsize=14)

plt.show()


import scipy.stats as stats
# ANOVA
result = stats.f_oneway(modelEquivalence['Log Likelihood'][modelEquivalence['Model'] == "UBVAE"],
                        modelEquivalence['Log Likelihood'][modelEquivalence['Model'] == "UBVAE 80"],
                        modelEquivalence['Log Likelihood'][modelEquivalence['Model'] == "UBVAE 60"],
                        modelEquivalence['Log Likelihood'][modelEquivalence['Model'] == "UBVAE 40"])
print(result)

res = stats.tukey_hsd(modelEquivalence['Log Likelihood'][modelEquivalence['Model'] == "UBVAE"],
                        modelEquivalence['Log Likelihood'][modelEquivalence['Model'] == "UBVAE 80"],
                        modelEquivalence['Log Likelihood'][modelEquivalence['Model'] == "UBVAE 60"],
                        modelEquivalence['Log Likelihood'][modelEquivalence['Model'] == "UBVAE 40"])
print(res)

assert(False)
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
#sns.set()
import scipy.stats as stats 

fig, axes = plt.subplots(nrows=1, ncols=2)


modelEquivalence = pd.read_pickle("./modelEquivalence.pkl")
modelEquivalence = modelEquivalence.reset_index()
modelEquivalence["Log Likelihood"] = -1 * modelEquivalence["Log Likelihood"] 
modelEquivalence = modelEquivalence.drop(columns=["level_0"])
modelEquivalence = modelEquivalence.drop(columns=["index"])


palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, n_colors=4) 
print(palette)


sns.barplot(data=modelEquivalence, x="Model", y="Log Likelihood", hue='Split', pallette=palette)
plt.title("Learned")
plt.show()


