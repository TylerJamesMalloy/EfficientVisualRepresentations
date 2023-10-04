import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

def aversion(coefficient, idf):
    errors = []
    for _, trial in idf.iterrows(): 
        if(trial['Trial Type'] == 'Utility Selection'):
            
            left_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", trial['Left Stimulus Marbles'])) 
            rght_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", trial['Right Stimulus Marbles'])) 
            left_marbles = list(int(a) for a in left_marbles)
            rght_marbles = list(int(a) for a in rght_marbles)
            # d = Left Stimulus, f = Right Stimulus
            chosen_ev = np.mean(left_marbles) if trial['Key Pressed'] == "d" else np.mean(rght_marbles)
            chosen_var = np.var(left_marbles) if trial['Key Pressed'] == "d" else np.var(rght_marbles)

            unchosen_ev = np.mean(rght_marbles) if trial['Key Pressed'] == "d" else np.mean(left_marbles)
            unchosen_var = np.var(rght_marbles) if trial['Key Pressed'] == "d" else np.var(left_marbles)

            chosen = chosen_ev - (coefficient * chosen_var)
            unchosen = unchosen_ev - (coefficient * unchosen_var)
            
            weighted = [chosen, unchosen]
            weighted = np.exp(weighted)/np.exp(weighted).sum()

            errors.append(weighted[1])

    return np.mean(errors) + np.abs(coefficient) * 1e-4

if __name__ == '__main__':
    df = pd.read_csv("../data/participantData.csv")
    #df = df[df['Experiment Type'] == 'Decision-Making']
    df = df[df['Experiment Type'] == 'Learning']
    ids = df["Id"].unique()

    participantAversionColumns = ["Id", "Risk Sensitivite Coefficient", "Type"]
    participantAversion = pd.DataFrame([], columns=participantAversionColumns)

    bnds = ((-1000, 1000),)
    changes = []

    for idx, id in enumerate(ids):
        #if(idx > 5): continue
        idf = df[df["Id"] == id]
        idf = idf.tail(180) # Skip learning trials

        #d = pd.DataFrame([[id, res.x[0], "Decision-Making"]], columns=participantAversionColumns) 
        #participantAversion = pd.concat([participantAversion, d])

        earlyTrials = idf.head(80)
        x0 = [0.0]
        res = sp.optimize.minimize(aversion, x0, args=(earlyTrials), method='Nelder-Mead', bounds=bnds, tol=1e-6)
        early_coef = res.x[0]

        d = pd.DataFrame([[id, early_coef, "Early Trials"]], columns=participantAversionColumns) 
        participantAversion = pd.concat([participantAversion, d])

        laterTrials = idf.tail(80)
        x0 = [0.0]
        res = sp.optimize.minimize(aversion, x0, args=(laterTrials), method='Nelder-Mead', bounds=bnds, tol=1e-6)
        later_coef = res.x[0]

        d = pd.DataFrame([[id, later_coef, "Later Trials"]], columns=participantAversionColumns) 
        participantAversion = pd.concat([participantAversion, d])

        change = np.abs(early_coef) - np.abs(later_coef) 
        print(change)
        changes.append(change)

    fig, axes = plt.subplots(nrows=2, ncols=1)

    early = participantAversion[participantAversion["Type"] == "Early Trials"]
    later = participantAversion[participantAversion["Type"] == "Later Trials"]

    sns.histplot(early, x="Risk Sensitivite Coefficient", ax=axes[0])
    sns.histplot(later, x="Risk Sensitivite Coefficient", ax=axes[1])
    plt.show()


    early = participantAversion[participantAversion["Type"] == "Early Trials"]
    sns.histplot(participantAversion, x="Risk Sensitivite Coefficient")
    plt.show()

    early = participantAversion[participantAversion["Type"] == "Later Trials"]
    sns.histplot(participantAversion, x="Risk Sensitivite Coefficient")
    plt.show()

    hist, bins = np.histogram(changes, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

    participantAversion.to_pickle("participantAversion_e2.pkl")