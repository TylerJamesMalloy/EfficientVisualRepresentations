import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 




reps = pd.read_pickle("Representations.pkl")
reps = reps.drop_duplicates()

marble_utilities = [3, 2, 2]

cols = ["Probability of Selection", "Outcome Variance Difference", "Beta"]
df = pd.DataFrame([
    [0.2, -0.5, 100],
    [0.4, -0.01, 100],
    [0.6, 0.5, 100],
    [0.2, 0.5, 4],
    [0.4, 0.01, 4],
    [0.6, -0.5, 4],
], columns=cols)

high = df[df["Beta"] == 100]
low = df[df["Beta"] == 4]

sns.regplot(high, x="Outcome Variance Difference", y="Probability of Selection", scatter=True, label="B=100", color="orange")
#sns.scatterplot(high, x="Outcome Variance Difference", y="Probability of Selection")

sns.regplot(low, x="Outcome Variance Difference", y="Probability of Selection", scatter=True, label="B=4", color="blue")
#sns.scatterplot(low, x="Outcome Variance Difference", y="Probability of Selection")
plt.show()

# Probability of Selection 
# Outcome Variance Difference 


print(reps)
assert(False)

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