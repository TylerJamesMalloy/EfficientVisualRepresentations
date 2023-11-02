import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 



fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True)

reps = pd.read_pickle("Representations.pkl")

sns.kdeplot(
    data=reps, x="Dimension 1", y="Dimension 2", hue="Utility",
    levels=5, thresh=.2, ax=axes[0]
)

reps = pd.read_pickle("Representations_Trained_u1.pkl")

sns.kdeplot(
    data=reps, x="Dimension 1", y="Dimension 2", hue="Utility",
    levels=5, thresh=.2, ax=axes[1]
)

reps = pd.read_pickle("Representations_Trained_u10.pkl")

sns.kdeplot(
    data=reps, x="Dimension 1", y="Dimension 2", hue="Utility",
    levels=5, thresh=.2, ax=axes[2]
)

reps = pd.read_pickle("Representations_Trained_u100.pkl")

sns.kdeplot(
    data=reps, x="Dimension 1", y="Dimension 2", hue="Utility",
    levels=5, thresh=.2, ax=axes[3]
)

plt.show()