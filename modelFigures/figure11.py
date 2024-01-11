import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
#sns.set()

fig, axes = plt.subplots(nrows=1, ncols=3)


modelFitAversion = pd.read_pickle("./modelFitChange.pkl")
palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, n_colors=4) 

modelBehaviorAversion = pd.read_pickle("./modelBehaviorChange.pkl")

sns.barplot(modelFitAversion, x="Model", y="Log Likelihood", hue="Split", palette=palette, ax=axes[2])

axes[2].set_ylabel("Log Likelihood", fontsize=12)
axes[2].set_xlabel("Data Split", fontsize=12)
axes[2].set_title("Log Likelihood by Data Split", fontsize=14)

group = modelBehaviorAversion.groupby(["Visual Difference"], as_index=False).agg({'Chosen':np.average})
group.reset_index(inplace=True)
sns.regplot(modelBehaviorAversion, x="Visual Difference", y="Chosen", scatter=False, label="U=100", ax=axes[1])
sns.scatterplot(group, x="Visual Difference", y="Chosen", ax=axes[1])

axes[1].set_ylabel("Model Probability Chosen", fontsize=12)
axes[1].set_xlabel("Visual Difference", fontsize=12)
axes[1].set_title("Model Probability Chosen by Visual Difference", fontsize=14)

group = modelBehaviorAversion.groupby(["Utility Difference"], as_index=False).agg({'Chosen':np.average})
group.reset_index(inplace=True)
sns.regplot(modelBehaviorAversion, x="Utility Difference", y="Chosen", scatter=False, label="U=100", ax=axes[0])
sns.scatterplot(group, x="Utility Difference", y="Chosen", ax=axes[0])

axes[0].set_ylabel("Model Probability Chosen", fontsize=12)
axes[0].set_xlabel("Utility Difference", fontsize=12)
axes[0].set_title("Model Probability Chosen by Utility Difference", fontsize=14)

plt.show()

