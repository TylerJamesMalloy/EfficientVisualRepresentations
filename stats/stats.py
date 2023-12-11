import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy import stats 
import numpy as np 

# participantFigures\fitAversion.pkl


participantAversion = pd.read_pickle("./participantAversion.pkl") # Experiment 1 decision-making 
participantChange = pd.read_pickle("./participantChange.pkl") # Experiment 1 change detection
participantLearned = pd.read_pickle("./participantLearned.pkl") # Experiment 2 decision-making
participantEquivalence = pd.read_pickle("./participantEquivalence.pkl") # Experiment 2 change detection

participantEquivalence.rename(columns={'Likelihood': 'Log Likelihood'}, inplace=True)
participantEquivalence.to_pickle("./participantEquivalence.pkl")

aversion = participantAversion.groupby(["Model", "Split"])["Log Likelihood"]
change = participantChange.groupby(["Model", "Split"])["Log Likelihood"]
learned = participantLearned.groupby(["Model", "Split"])["Log Likelihood"]
equivalence = participantEquivalence.groupby(["Model", "Split"])["Log Likelihood"]

aversion_loss_mean = aversion.mean().to_numpy()
change_loss_mean = change.mean().to_numpy()
learned_loss_mean = learned.mean().to_numpy()
equivalence_loss_mean = equivalence.mean().to_numpy()

aversion_percentage = 1 - np.exp(-1 * aversion_loss_mean)
change_percentage = 1 - np.exp(-1 * change_loss_mean)
learned_percentage = 1 - np.exp(-1 * learned_loss_mean)
equivalence_percentage = 1 - np.exp(-1 * equivalence_loss_mean)

aversion_loss_var = aversion.var().to_numpy()
change_loss_var = change.var().to_numpy()
learned_loss_var = learned.var().to_numpy()
equivalence_loss_var = equivalence.var().to_numpy()

obs = 200
paics = []
print(change_percentage)
for i in range(len(aversion_percentage)):
    print("Participant Aversion Accuracy: ", aversion_percentage[i], " +\-", aversion_loss_var[i])
    print("Participant Change Accuracy: ", change_percentage[i], " +\-", change_loss_var[i])
    print("Participant Learned Accuracy: ", learned_percentage[i], " +\-",learned_loss_var[i])
    print("Participant Equivalence Accuracy: ", equivalence_percentage[i], " +\-", equivalence_loss_var[i])
    if(i < 4):
        n = 145
        #print("Participant Aversion BIC: ", 2 - 2*np.log(1- aversion_percentage[i]))
        #paics.append(round(2*np.log(n) - 2*(np.log(aversion_percentage[i])* n) ,2))
        #print("Participant Change BIC: ", 2 - 2*np.log( change_percentage[i]))
        #paics.append(round(2*np.log(n) - 2*(np.log(change_percentage[i])* n) ,2))
        n = 133
        #print("Participant Learned BIC: ", 4 - 2*np.log(learned_percentage[i]))
        #paics.append(round(4*np.log(n) - 2*(np.log(learned_percentage[i])* n) ,2))
        #print("Participant Equivalence BIC: ", 4 - 2*np.log( equivalence_percentage[i]))
        #paics.append(round(4*np.log(n) - 2*(np.log(equivalence_percentage[i])* n) ,2))
    
#print(paics)

"""
Participant Aversion Accuracy:  0.8364335090820032  +\- 3.8089672743141194
Participant Change Accuracy:  0.7754899296868045  +\- 0.19034331970902027
Participant Learned Accuracy:  0.6275379786883329  +\- 0.04628593865240477
Participant Equivalence Accuracy:  0.9786457570853309  +\- 10.910304535167828

Participant Aversion Accuracy:  0.7418609428131138  +\- 2.5969000941500675
Participant Change Accuracy:  0.7791638167687477  +\- 0.16974685024604688
Participant Learned Accuracy:  0.4254187596621495  +\- 0.04378098662614882
Participant Equivalence Accuracy:  0.6690586477005638  +\- 3.072885838971045

Participant Aversion Accuracy:  0.7396796845381617  +\- 2.791746836996719
Participant Change Accuracy:  0.7881973302925193  +\- 0.40436068431772637
Participant Learned Accuracy:  0.49722040128857004  +\- 0.043091507282356944
Participant Equivalence Accuracy:  0.9135604622015985  +\- 8.062540257441277

Participant Aversion Accuracy:  0.794281596137294  +\- 3.8883013664962744
Participant Change Accuracy:  0.8025488164661597  +\- 0.4321073815176503
Participant Learned Accuracy:  0.5473783112576568  +\- 0.03925718670515891
Participant Equivalence Accuracy:  0.9638676426930209  +\- 16.955057697172258

Participant Aversion Accuracy:  0.5232551339483449  +\- 0.030316154673321313
Participant Change Accuracy:  0.737583770339272  +\- 0.033030493740302716
Participant Learned Accuracy:  0.6969441134517607  +\- 1.2286332055065934e-05
Participant Equivalence Accuracy:  0.707056536573516  +\- 2.300305698897666e-05
"""


modelFitAversion = pd.read_pickle("./modelFitAversion.pkl") # Experiment 1 decision-making 
modelFitChange = pd.read_pickle("./modelFitChange.pkl") # Experiment 1 change detection
modelFitLearned = pd.read_pickle("./modelFitEquivalence.pkl") # Experiment 2 change detection 
modelFitEquivalence = pd.read_pickle("./modelFitLearned.pkl") # Experiment 2 decision-making

aversion = modelFitAversion.groupby(["Model"])["Log Likelihood"]
change = modelFitChange.groupby(["Model"])["Log Likelihood"]
learned = modelFitLearned.groupby(["Model"])["Log Likelihood"]
equivalence = modelFitEquivalence.groupby(["Model"])["Log Likelihood"]


aversion_loss_mean = aversion.mean().to_numpy()
change_loss_mean = change.mean().to_numpy()
learned_loss_mean = learned.mean().to_numpy()
equivalence_loss_mean = equivalence.mean().to_numpy()

aversion_percentage = 1 - np.exp(-1 * aversion_loss_mean)
change_percentage = 1 - np.exp(-1 * change_loss_mean)
learned_percentage = 1 - np.exp(-1 * learned_loss_mean)
equivalence_percentage = 1 - np.exp(-1 * equivalence_loss_mean)

aversion_loss_var = aversion.var().to_numpy()
change_loss_var = change.var().to_numpy()
learned_loss_var = learned.var().to_numpy()
equivalence_loss_var = equivalence.var().to_numpy()

#change_percentage = [.874, .873, .870, 0.873]
#learned_percentage = [.657, 0.655, 0.648, 0.642]
aics = []
for i in range(len(aversion_percentage)):
    print("Model Aversion Accuracy: ", aversion_percentage[i], " +\-", aversion_loss_var[i])
    print("Model Change Accuracy: ", change_percentage[i], " +\-", change_loss_var[i])
    print("Model Learned Accuracy: ", learned_percentage[i], " +\-",learned_loss_var[i])
    print("Model Equivalence Accuracy: ", equivalence_percentage[i], " +\-", equivalence_loss_var[i])

    n = 145
    #print("Model Aversion BIC: ", 4*np.log(n) - 2*(np.log(aversion_percentage[i])* n))
    #aics.append(round(4*np.log(n) - 2*(np.log(aversion_percentage[i])* n) ,2))
    #print("Model Change BIC: ", 4*np.log(n) - 2*(np.log(change_percentage[i])* n))
    #aics.append(round(4*np.log(n) - 2*(np.log(change_percentage[i])* n) ,2))
    n = 133 
    #print("Model Learned BIC: ", 4*np.log(n) - 2*(np.log(learned_percentage[i])* n))
    #aics.append(round(4*np.log(n) - 2*(np.log(learned_percentage[i])* n) ,2))
    #print("Model Equivalence BIC: ", *np.log(n) - 2*(np.log(equivalence_percentage[i])* n))
    #aics.append(round(4*np.log(n) - 2*(np.log(equivalence_percentage[i])* n) ,2))
    
#print(aics)

paics = np.asarray(paics)
aics = np.asarray(aics)

#print(np.sum(aics < paics))

#print(aics - paics)

"""
Model Equivalence Accuracy:  0.8745523150773342  +\- 0.08219877321560742
Model Equivalence Accuracy:  0.8683747665349526  +\- 0.10952408811934926
Model Equivalence Accuracy:  0.8708239433359429  +\- 0.1419698272551657
Model Equivalence Accuracy:  0.8731942995984106  +\- 0.2074769709384845
Model Aversion Accuracy:  0.9563540548046604  +\- 0.40259002491129725
Model Change Accuracy:  0.6479517467824469  +\- 0.06765194019135296
Model Learned Accuracy:  0.9563271194243628  +\- 0.40206739692570825
Model Aversion Accuracy:  0.9480777874505776  +\- 0.7258415694817433
Model Change Accuracy:  0.6278426816497533  +\- 0.09233174745326354
Model Learned Accuracy:  0.9480457823277693  +\- 0.7250696703881923
Model Aversion Accuracy:  0.9504443994612242  +\- 0.896003668752981
Model Change Accuracy:  0.6324919995453713  +\- 0.11866303107174933
Model Learned Accuracy:  0.9504144442496446  +\- 0.8954977440031688
Model Aversion Accuracy:  0.945641382028871  +\- 1.3086374187133563
Model Change Accuracy:  0.6447933777579711  +\- 0.1827857806014403
Model Learned Accuracy:  0.9456048063738489  +\- 1.3076841966115982
"""



# Calculate AIC 


"""for df in [modelFitAversion, modelEquivalence, modelFitChange]:
    corrs = []
    for id in df["Id"].unique():
        pfit = df[df["Id"] == id]
        corr = 2 / (len(pfit["Beta"].unique()) + len(pfit["Upsilon"].unique()))
        corrs.append(corr)

    print(np.mean(corrs))
    print(np.var(corrs))
"""

"""
participantChange.rename(columns={'Likelihood': 'Log Likelihood'}, inplace=True)

participantChange.loc[participantChange["Model"] == "Utility 60", 'Model'] = "Utility"
participantChange.loc[participantChange["Model"] == "Utility 80", 'Model'] = "Utility"
participantChange.loc[participantChange["Model"] == "Utility 40", 'Model'] = "Utility"

participantChange['Log Likelihood'] = -1 * np.log( participantChange['Log Likelihood'])

participantChange.to_pickle("./participantChange.pkl")
#participantFitLearned.rename(columns={'Likelihood': 'Log Likelihood'}, inplace=True)

participantChange = pd.read_pickle("./participantChange.pkl")
"""