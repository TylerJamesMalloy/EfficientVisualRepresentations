import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
#sns.set()

fig, axes = plt.subplots(nrows=1, ncols=2)


modelFitAversion = pd.read_pickle("./modelFitAversion.pkl")
palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, n_colors=4) 

sns.barplot(modelFitAversion, x="Model", y="Log Likelihood", hue="Split", palette=palette, ax=axes[1])

axes[1].set_xlabel("Data Split", fontsize=12)
axes[1].set_ylabel("Model Log Likelihood", fontsize=12)
axes[1].set_title("Model Log Likelihood by Data Split", fontsize=14)

modelBehaviorAversion_Low = pd.read_pickle("./modelBehaviorAversion_Low.pkl")
modelBehaviorAversion_Low = modelBehaviorAversion_Low.round(1)

group = modelBehaviorAversion_Low.groupby(["Outcome Variance Difference"], as_index=False).agg({'Chosen':np.average})
group.reset_index(inplace=True)
sns.regplot(modelBehaviorAversion_Low, x="Outcome Variance Difference", y="Chosen", scatter=False, label="B=4", color="blue", ax=axes[0])
sns.scatterplot(group, x="Outcome Variance Difference", y="Chosen", ax=axes[0])

modelBehaviorAversion_High = pd.read_pickle("./modelBehaviorAversion_High.pkl")
modelBehaviorAversion_High = modelBehaviorAversion_High.round(1)

group = modelBehaviorAversion_High.groupby(["Outcome Variance Difference"], as_index=False).agg({'Chosen':np.average})
group.reset_index(inplace=True)
sns.regplot(modelBehaviorAversion_High, x="Outcome Variance Difference", y="Chosen", scatter=False, label="B=100", color="orange", ax=axes[0])
sns.scatterplot(group, x="Outcome Variance Difference", y="Chosen", ax=axes[0])

axes[0].set_ylabel("Model Probability of Selection", fontsize=12)
axes[0].set_xlabel("Outcome Variance Difference", fontsize=12)
axes[0].set_title("Model Probability of Selection by Outcome Variance", fontsize=14)

#sns.barplot(modelAversion, x="Model", y="Log Likelihood", hue="Split")
plt.show()