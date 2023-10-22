import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

def aversion(coefficient, idf, test=False):
    nll = []
    accuracy = []
    aversion_coefficinet = coefficient[0]
    learning_rate = coefficient[1]
    old_block = 1.0
    num_util_observations = 0

    for _, trial in idf.iterrows():
        if(trial['Block'] != old_block):
            old_block = trial['Block'] 
            num_util_observations = 0

        if(trial['Trial Type'] == 'Utility Selection'):
            num_util_observations += 1
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

            learned_coefficient = aversion_coefficinet * (learning_rate * (np.log(num_util_observations + 1) / np.log(10)))
            #if(chosen_ev == unchosen_ev): continue 
            chosen = chosen_ev - (learned_coefficient * chosen_var)
            unchosen = unchosen_ev - (learned_coefficient * unchosen_var)
            
            weighted = np.array([chosen, unchosen])
            weighted = np.exp(weighted)/np.exp(weighted).sum()

            accuracy.append(weighted[0])
            # Negative Log Loss
            loss = np.log(weighted[1])
            #loss = -1 * (1*np.log10(p))
            nll.append(loss)
    
    if(test):
        return np.mean(nll)
    else:
        return np.mean(nll) #(-1 * np.mean(accuracy)) + np.abs(coefficient) * 1e-3

if __name__ == '__main__':
    df = pd.read_csv("../data/participantData.csv")
    df = df[df['Experiment Type'] == 'Learning']
    ids = df["Id"].unique()

    learnedAversionColumns = ["Id", "Model", "Split", "Log Likelihood", "Aversion Coefficient", "Learning Rate"]
    learnedAversion = pd.DataFrame([], columns=learnedAversionColumns)

    
    changes = []
    x0 = [0.0, 0.5]

    good = []

    for idx, id in enumerate(ids):
        idf = df[df["Id"] == id]
        idf = idf.tail(180) # Skip learning trials
        
        rng = np.random.default_rng(id)
        EUT_Loss = aversion((0,0), idf, True)
        #good.append(id)

        d = pd.DataFrame([[id, "EUT", 100, EUT_Loss, 0, 0]], columns=learnedAversionColumns) 
        learnedAversion = pd.concat([learnedAversion, d])

        bnds = ((-100, 100), (0, 1),)

        res = sp.optimize.minimize(aversion, x0, args=(idf), method='L-BFGS-B', bounds=bnds, tol=1e-10)
        aversion_coefficinet = res.x[0]
        learning_rate = res.x[1] 

        rng = np.random.default_rng(id)

        CPT_Loss = aversion((aversion_coefficinet, learning_rate), idf, True)

        d = pd.DataFrame([[id, "CPT", 100, CPT_Loss, aversion_coefficinet, learning_rate]], columns=learnedAversionColumns) 
        learnedAversion = pd.concat([learnedAversion, d])

        msk = rng.random(len(idf)) < 0.80
        train = idf[msk]
        
        res = sp.optimize.minimize(aversion, x0, args=(train), method='L-BFGS-B', bounds=bnds, tol=1e-10)
        aversion_coefficinet = res.x[0]
        learning_rate = res.x[1] 

        test = idf[~msk]

        Loss = aversion((aversion_coefficinet, learning_rate), test, True)
        
        d = pd.DataFrame([[id, "CPT", 80, Loss, aversion_coefficinet, learning_rate]], columns=learnedAversionColumns) 
        learnedAversion = pd.concat([learnedAversion, d])

        msk = rng.random(len(idf)) < 0.60
        train = idf[msk]
        
        res = sp.optimize.minimize(aversion, x0, args=(train), method='L-BFGS-B', bounds=bnds, tol=1e-10)
        aversion_coefficinet = res.x[0]
        learning_rate = res.x[1] 

        test = idf[~msk]

        Loss = aversion((aversion_coefficinet, learning_rate), test, True)
        
        d = pd.DataFrame([[id, "CPT", 60, Loss, aversion_coefficinet, learning_rate]], columns=learnedAversionColumns) 
        learnedAversion = pd.concat([learnedAversion, d])

        msk = rng.random(len(idf)) < 0.40
        train = idf[msk]
        
        res = sp.optimize.minimize(aversion, x0, args=(train), method='L-BFGS-B', bounds=bnds, tol=1e-10)
        aversion_coefficinet = res.x[0]
        learning_rate = res.x[1] 

        test = idf[~msk]

        Loss = aversion((aversion_coefficinet, learning_rate), test, True)
        
        d = pd.DataFrame([[id, "CPT", 40, Loss, aversion_coefficinet, learning_rate]], columns=learnedAversionColumns) 
        learnedAversion = pd.concat([learnedAversion, d])

    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.barplot(learnedAversion, x="Model", y="Log Likelihood", hue="Split", ax=axes[0])

    learnedAversion.to_pickle("./fitLearned.pkl")
    
    plt.show()

    