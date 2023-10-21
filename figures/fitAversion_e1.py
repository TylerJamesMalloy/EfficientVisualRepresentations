import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

def aversion(coefficient, idf, test=False):
    nll = []
    accuracy = []
    for _, trial in idf.iterrows(): 
        if(trial['Trial Type'] == 'Utility Selection'):
            
            left_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", trial['Left Stimulus Marbles'])) 
            rght_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", trial['Right Stimulus Marbles'])) 
            left_marbles = list(int(a) for a in left_marbles)
            rght_marbles = list(int(a) for a in rght_marbles)
            # d = Left Stimulus, f = Right Stimulus
            if(trial['Key Pressed'] != "d" and trial['Key Pressed'] != "f"): continue 
            chosen_ev = np.mean(left_marbles) if trial['Key Pressed'] == "d" else np.mean(rght_marbles)
            chosen_var = np.var(left_marbles) if trial['Key Pressed'] == "d" else np.var(rght_marbles)

            unchosen_ev = np.mean(rght_marbles) if trial['Key Pressed'] == "d" else np.mean(left_marbles)
            unchosen_var = np.var(rght_marbles) if trial['Key Pressed'] == "d" else np.var(left_marbles)

            #if(chosen_ev == unchosen_ev): continue 
            chosen = chosen_ev - (coefficient * chosen_var)
            unchosen = unchosen_ev - (coefficient * unchosen_var)
            
            weighted = np.array([chosen, unchosen])
            #weighted *= 2
            weighted = np.exp(weighted)/np.exp(weighted).sum()

            accuracy.append(weighted[0])
            # Negative Log Loss
            loss = np.log(weighted[1])
            #loss = -1 * (1*np.log10(p))
            nll.append(loss)
    
    if(test):
        return np.mean(loss)
    else:
        return (-1 * np.mean(accuracy)) + np.abs(coefficient) * 1e-3

if __name__ == '__main__':
    df = pd.read_csv("../data/participantData.csv")
    df = df[df['Experiment Type'] == 'Decision-Making']
    #df = df[df['Experiment Type'] == 'Learning']
    ids = df["Id"].unique()

    participantAversionColumns = ["Id", "Model", "Split", "Log Likelihood", "Coefficient"]
    participantAversion = pd.DataFrame([], columns=participantAversionColumns)

    
    changes = []
    x0 = [0.0]

    good = []

    for idx, id in enumerate(ids):
        idf = df[df["Id"] == id]
        idf = idf.tail(180) # Skip learning trials
        #if(idx > 40): continue 
        
        rng = np.random.default_rng(id)

        EUT_Loss = aversion(0, idf, True)
        
        good.append(id)

        d = pd.DataFrame([[id, "EUT", 100, EUT_Loss, 0]], columns=participantAversionColumns) 
        participantAversion = pd.concat([participantAversion, d])

        bnds = ((-100, 100), )

        res = sp.optimize.minimize(aversion, x0, args=(idf), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        coef = res.x[0]
        
        

        rng = np.random.default_rng(id)

        CPT_Loss = aversion(coef, idf, True)

        d = pd.DataFrame([[id, "CPT", 100, CPT_Loss, coef]], columns=participantAversionColumns) 
        participantAversion = pd.concat([participantAversion, d])


        msk = rng.random(len(idf)) < 0.80
        train = idf[msk]
        
        res = sp.optimize.minimize(aversion, x0, args=(train), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        coef = res.x[0]
        test = idf[~msk]

        Loss = aversion(coef, test, True)
        
        d = pd.DataFrame([[id, "CPT", 20, Loss, coef]], columns=participantAversionColumns) 
        participantAversion = pd.concat([participantAversion, d])

        msk = rng.random(len(idf)) < 0.60
        train = idf[msk]
        
        res = sp.optimize.minimize(aversion, x0, args=(train), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        coef = res.x[0]

        test = idf[~msk]

        Loss = aversion(coef, test, True)
        
        d = pd.DataFrame([[id, "CPT", 40, Loss, coef]], columns=participantAversionColumns) 
        participantAversion = pd.concat([participantAversion, d])

        msk = rng.random(len(idf)) < 0.40
        train = idf[msk]
        
        res = sp.optimize.minimize(aversion, x0, args=(train), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        coef = res.x[0]

        test = idf[~msk]

        Loss = aversion(coef, test, True)
        
        d = pd.DataFrame([[id, "CPT", 60, Loss, coef]], columns=participantAversionColumns) 
        participantAversion = pd.concat([participantAversion, d])

    fig, axes = plt.subplots(nrows=1, ncols=2)

    participantAversion.to_pickle(".\splitAversion_e1.pkl")

    splitAversion = pd.read_pickle(".\splitAversion_e1.pkl")
    splitAversion = splitAversion.reset_index()
    #splitAversion = splitAversion[splitAversion['Risk Sensitivite Coefficient'] > -50]
    #splitAversion = splitAversion[splitAversion['Risk Sensitivite Coefficient'] < 50]
    #splitAversion["Accuracy"] *= -1
    sns.barplot(splitAversion, x="Model", y="Log Likelihood", hue="Split", ax=axes[0])

    axes[0].set_title("Log Likelihood by Model and Split")
    axes[1].set_title("Risk Coefficients 100% Fit")

    splitAversion = splitAversion[splitAversion["Model"] == "CPT"]
    splitAversion = splitAversion[splitAversion["Split"] == 100]
    sns.histplot(splitAversion, x="Coefficient", ax=axes[1])
    plt.show()


    