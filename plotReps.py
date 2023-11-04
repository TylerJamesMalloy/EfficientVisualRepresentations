import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 



fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True)

reps = pd.read_pickle("Representations.pkl")

for col, upsilon in enumerate([0, 1e6]):
    for row, beta in enumerate([0, 10]):
        plot_reps = reps[(reps["Beta"] == beta) & (reps["Upsilon"] == upsilon)]
        print(plot_reps)
        sns.kdeplot(
            data=plot_reps, x="Dimension 1", y="Dimension 2", hue="Utility",
            levels=5, thresh=.2, ax=axes[row, col]
        )


plt.show()

"""
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 



fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True)

reps = pd.read_pickle("Representations.pkl")
print(reps["Beta"].unique())
print(reps["Upsilon"].unique())

for col, upsilon in enumerate([0, 100]):
    for row, beta in enumerate([0, 100]):
        plot_reps = reps[(reps["Beta"] == beta) & (reps["Upsilon"] == upsilon)]
        print(plot_reps)
        sns.kdeplot(
            data=plot_reps, x="Dimension 1", y="Dimension 2", hue="Utility",
            levels=5, thresh=.2, ax=axes[row, col]
        )

plt.show()
"""