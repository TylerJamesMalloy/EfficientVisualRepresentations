import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy_indexed as npi

def aversion(coefficients, idf, test=False):
    utility_coefficient = coefficients[0]
    learning_rate = coefficients[1]
    old_block = 1.0
    num_util_observations = 0
    Losses = []
    Accuracies = []

    for _, trial in idf.iterrows(): 
        if(trial['Block'] != old_block):
            old_block = trial['Block'] 
            num_util_observations = 0

        if(trial['Trial Type'] == 'Utility Selection'):
            num_util_observations += 1

        if(trial['Trial Type'] == 'Change Detection' and trial['Changed']):
            
            original_marbles = trial['Right Stimulus Marbles'] if trial['Changed Index'] else trial['Left Stimulus Marbles']
            new_marbles = trial['New Stimulus Marbles'] 
            original_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", original_marbles)) 
            new_marbles = list(filter(lambda a: a != "[" and a != "]" and a != " " and a != ",", new_marbles)) 
            original_marbles = list(int(a) for a in original_marbles)
            new_marbles = list(int(a) for a in new_marbles)

            original_ev = np.mean(original_marbles)
            new_ev = np.mean(new_marbles)

            coefficient = utility_coefficient * (learning_rate * (num_util_observations / 10))

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

    fitEquivalenceColumns = ["Id", "Model", "Split", "Likelihood", "Utility Coefficient", "Learning Rate"]
    fitEquivalence = pd.DataFrame([], columns=fitEquivalenceColumns)

    
    changes = []
    x0 = [0.0, 0.5]

    good = []

    for idx, id in enumerate(ids):
        idf = df[df["Id"] == id]
        idf = idf.tail(180) # Skip learning trials
        #if(idx > 40): continue 
        
        rng = np.random.default_rng(id)
        EUT_Loss = -1 * aversion((0,0), idf, True)
        
        good.append(id)

        d = pd.DataFrame([[id, "Visual", 100, EUT_Loss, 0, 0]], columns=fitEquivalenceColumns) 
        fitEquivalence = pd.concat([fitEquivalence, d])

        bnds = ((-100, 100), (0, 100))

        res = sp.optimize.minimize(aversion, x0, args=(idf), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        utility_coef = res.x[0]
        learning_rate = res.x[1]
        
        rng = np.random.default_rng(id)

        CPT_Loss = -1 * aversion((utility_coef, learning_rate), idf, True)

        d = pd.DataFrame([[id, "Utility", 100, CPT_Loss, utility_coef, learning_rate]], columns=fitEquivalenceColumns) 
        fitEquivalence = pd.concat([fitEquivalence, d])


        msk = rng.random(len(idf)) < 0.80
        train = idf[msk]
        
        res = sp.optimize.minimize(aversion, x0, args=(train), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        utility_coef = res.x[0]
        learning_rate = res.x[1]
        test = idf[~msk]

        Loss = -1 * aversion((utility_coef, learning_rate), test, True)
        
        d = pd.DataFrame([[id, "Utility", 80, Loss, utility_coef, learning_rate]], columns=fitEquivalenceColumns) 
        fitEquivalence = pd.concat([fitEquivalence, d])

        msk = rng.random(len(idf)) < 0.60
        train = idf[msk]
        
        res = sp.optimize.minimize(aversion, x0, args=(train), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        utility_coef = res.x[0]
        learning_rate = res.x[1]

        test = idf[~msk]

        Loss = -1 * aversion((utility_coef, learning_rate), test, True)
        
        d = pd.DataFrame([[id, "Utility", 60, Loss, utility_coef, learning_rate]], columns=fitEquivalenceColumns) 
        fitEquivalence = pd.concat([fitEquivalence, d])

        msk = rng.random(len(idf)) < 0.40
        train = idf[msk]
        
        res = sp.optimize.minimize(aversion, x0, args=(train), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        utility_coef = res.x[0]
        learning_rate = res.x[1]

        test = idf[~msk]

        Loss = -1 * aversion((utility_coef, learning_rate), test, True)
        
        d = pd.DataFrame([[id, "Utility", 40, Loss, utility_coef, learning_rate]], columns=fitEquivalenceColumns) 
        fitEquivalence = pd.concat([fitEquivalence, d])

    fitEquivalence.to_pickle("./fitEquivalence.pkl")

    fitEquivalence = pd.read_pickle("./fitEquivalence.pkl")
    fitEquivalence = fitEquivalence.reset_index()

    order = ["Visual", "Utility 40", "Utility 60", "Utility 80"]

    fitEquivalence.loc[fitEquivalence['Split'] == 80, 'Model'] = "Utility 80"
    fitEquivalence.loc[fitEquivalence['Split'] == 60, 'Model'] = "Utility 60"
    fitEquivalence.loc[fitEquivalence['Split'] == 40, 'Model'] = "Utility 40"

    print(fitEquivalence.groupby["Model"].means())

    sns.barplot(data=fitEquivalence, x="Model", y="Likelihood", errorbar=('ci', 90), hue="Split")
    #plt.ylim(0.7, 0.825)
    plt.show()


    