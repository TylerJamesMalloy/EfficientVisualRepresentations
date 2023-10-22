import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy_indexed as npi

def aversion(coefficient, idf, test=False):
    Losses = []
    Accuracies = []
    for _, trial in idf.iterrows(): 
        if(trial['Trial Type'] == 'Change Detection' and trial['Changed']):
            
            original_marbles = trial['Right Stimulus Marbles'] if trial['Changed Index'] else trial['Left Stimulus Marbles']
            new_marbles = trial['New Stimulus Marbles'] 
            original_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", original_marbles)) 
            new_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", new_marbles)) 
            original_marbles = list(int(a) for a in original_marbles)
            new_marbles = list(int(a) for a in new_marbles)

            original_ev = np.mean(original_marbles)
            new_ev = np.mean(new_marbles)

            max_diff = (2 - 4) ** 2
            ev_diff = ((original_ev - new_ev) ** 2) / max_diff
            vis_diff = np.sum([original_marbles[x] != new_marbles[x] for x in range(len(original_marbles))]) / 9
            weighted_diff = (((1 - coefficient) * vis_diff) + (coefficient * ev_diff)) 
            weighted = [1 - weighted_diff, weighted_diff] #[No change detected, change detected]
            weighted = np.exp(weighted)/np.exp(weighted).sum()

            correct = trial['Correct']
            incorrect = 0 if trial['Correct'] else 1
            Losses.append(np.log(weighted[incorrect]))
            Accuracies.append(weighted[correct])
            
    
    if(test):
        return np.mean(Losses)
    else:
        return (-1 * np.mean(Accuracies)) #+ np.abs(coefficient) * 1e-3

if __name__ == '__main__':
    df = pd.read_csv("../data/participantData.csv")
    df = df[df['Experiment Type'] == 'Decision-Making']
    #df = df[df['Experiment Type'] == 'Learning']
    ids = df["Id"].unique()

    participantChangeColumns = ["Id", "Model", "Split", "Likelihood", "Coefficient"]
    participantChange = pd.DataFrame([], columns=participantChangeColumns)

    
    changes = []
    x0 = [0.0]

    good = []

    for idx, id in enumerate(ids):
        idf = df[df["Id"] == id]
        idf = idf.tail(180) # Skip learning trials
        #if(idx > 40): continue 
        
        rng = np.random.default_rng(id)

        EUT_Loss = -1 * aversion(0, idf, True)
        
        good.append(id)

        d = pd.DataFrame([[id, "Visual", 100, EUT_Loss, 0]], columns=participantChangeColumns) 
        participantChange = pd.concat([participantChange, d])

        bnds = ((-100, 100), )

        res = sp.optimize.minimize(aversion, x0, args=(idf), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        coef = res.x[0]
        
        rng = np.random.default_rng(id)

        CPT_Loss = -1 * aversion(coef, idf, True)

        d = pd.DataFrame([[id, "Utility", 100, CPT_Loss, coef]], columns=participantChangeColumns) 
        participantChange = pd.concat([participantChange, d])


        msk = rng.random(len(idf)) < 0.80
        train = idf[msk]
        
        res = sp.optimize.minimize(aversion, x0, args=(train), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        coef = res.x[0]
        test = idf[~msk]

        Loss = -1 * aversion(coef, test, True)
        
        d = pd.DataFrame([[id, "Utility", 80, Loss, coef]], columns=participantChangeColumns) 
        participantChange = pd.concat([participantChange, d])

        msk = rng.random(len(idf)) < 0.60
        train = idf[msk]
        
        res = sp.optimize.minimize(aversion, x0, args=(train), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        coef = res.x[0]

        test = idf[~msk]

        Loss = -1 * aversion(coef, test, True)
        
        d = pd.DataFrame([[id, "Utility", 60, Loss, coef]], columns=participantChangeColumns) 
        participantChange = pd.concat([participantChange, d])

        msk = rng.random(len(idf)) < 0.40
        train = idf[msk]
        
        res = sp.optimize.minimize(aversion, x0, args=(train), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        coef = res.x[0]

        test = idf[~msk]

        Loss = -1 * aversion(coef, test, True)
        
        d = pd.DataFrame([[id, "Utility", 40, Loss, coef]], columns=participantChangeColumns) 
        participantChange = pd.concat([participantChange, d])

    participantChange.to_pickle(".\splitChange_2.pkl")

    splitChange = pd.read_pickle(".\splitChange_2.pkl")
    splitChange = splitChange.reset_index()

    order = ["Visual", "Utility 40", "Utility 60", "Utility 80"]

    splitChange.loc[splitChange['Split'] == 80, 'Model'] = "Utility 80"
    splitChange.loc[splitChange['Split'] == 60, 'Model'] = "Utility 60"
    splitChange.loc[splitChange['Split'] == 40, 'Model'] = "Utility 40"

    sns.barplot(data=splitChange, x="Model", y="Likelihood", errorbar=('ci', 90), hue="Split")
    #plt.ylim(0.7, 0.825)
    plt.show()


    