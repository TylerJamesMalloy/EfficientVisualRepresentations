import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy import stats 
import numpy as np 


modelFitAversion = pd.read_pickle("./modelFigures/modelFitAversion.pkl")
modelEquivalence = pd.read_pickle("./modelFigures/modelEquivalence.pkl")
modelFitChange = pd.read_pickle("./modelFigures/modelFitChange.pkl")

for df in [modelFitAversion, modelEquivalence, modelFitChange]:
    corrs = []
    for id in df["Id"].unique():
        pfit = df[df["Id"] == id]
        corr = 2 / (len(pfit["Beta"].unique()) + len(pfit["Upsilon"].unique()))
        corrs.append(corr)

    print(np.mean(corrs))
    print(np.var(corrs))


