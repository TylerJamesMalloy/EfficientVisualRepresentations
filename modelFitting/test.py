import scipy as sp
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

def Accuracy(coefficient, model, idf, test=False):
    errors = []
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

            if(chosen_ev == unchosen_ev): continue 

            chosen = chosen_ev - (coefficient * chosen_var)
            unchosen = unchosen_ev - (coefficient * unchosen_var)
            
            weighted = [chosen, unchosen]
            weighted = np.exp(weighted)/np.exp(weighted).sum()

            errors.append(weighted[1])
    
    if(test):
        return np.mean(errors)
    else:
        return np.mean(errors) + np.abs(coefficient) * 1e-4

if __name__ == '__main__':
    df = pd.read_csv("../data/participantData.csv")
    df = df[df['Experiment Type'] == 'Decision-Making']
    #df = df[df['Experiment Type'] == 'Learning']
    ids = df["Id"].unique()

    participantAccuracyColumns = ["Id", "Model", "Type", "Accuracy"]
    participantAccuracy = pd.DataFrame([], columns=participantAccuracyColumns)

    models = []
    for stimuli_set in [0,1,2,3,4,5]:
        model = load_model(exp_dir + "/set" + str(stimuli_set))
        model.to(device)

        #gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
        train_loader = get_dataloaders(args.dataset,
                                        batch_size=args.batch_size,
                                        logger=logger,
                                        set=stimuli_set)
        
        logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        loss_f = get_loss_f(args.loss,
                            n_data=len(train_loader.dataset),
                            device=device,
                            **vars(args))
        trainer = Trainer(model, optimizer, loss_f,
                            device=device,
                            logger=None,
                            save_dir=exp_dir + "/set" + str(stimuli_set),
                            is_progress_bar=not args.no_progress_bar)
        
        # fit utilities to participant? 
        # LOO based? 
        utilities = np.array(stimuli_mean_utilities)
        utilities = torch.from_numpy(utilities.astype(np.float64)).float()
        trainer(train_loader,
                utilities=utilities, 
                epochs=1,
                checkpoint_every=args.checkpoint_every,)

        models.append(model)

    bnds = ((-100, 100),)
    changes = []

    for idx, id in enumerate(ids):
        np.random.seed(id)
        #if(idx > 5): continue
        idf = df[df["Id"] == id]
        idf = idf.tail(180) # Skip learning trials

        x0 = [0.0]
        res = sp.optimize.minimize(Accuracy, x0, args=(model, idf), method='L-BFGS-B', bounds=bnds, tol=1e-6)
        coef = res.x[0]

        #print(res.fun)
        accuracy = 1 - (Accuracy(coef, idf, True))

        d = pd.DataFrame([[id, coef, "UB-VAE", accuracy]], columns=participantAccuracyColumns) 
        participantAccuracy = pd.concat([participantAccuracy, d])


        msk = np.random.rand(len(idf)) < 0.8
        train = idf[msk]
        
        x0 = [0.0]
        res = sp.optimize.minimize(Accuracy, x0, args=(model, train), method='L-BFGS-B', bounds=bnds, tol=1e-6)
        split_coef = res.x[0]

        test = idf[~msk]

        accuracy = 1 - (Accuracy(split_coef, test, True))
        
        d = pd.DataFrame([[id, coef, "UB-VAE", accuracy]], columns=participantAccuracyColumns) 
        participantAccuracy = pd.concat([participantAccuracy, d])

        msk = np.random.rand(len(idf)) < 0.5
        train = idf[msk]
        
        x0 = [0.0]
        res = sp.optimize.minimize(Accuracy, x0, args=(model, train), method='L-BFGS-B', bounds=bnds, tol=1e-6)
        split_coef = res.x[0]

        test = idf[~msk]

        accuracy = 1 - (Accuracy(split_coef, test, True))
        
        d = pd.DataFrame([[id, coef, "UB-VAE", accuracy]], columns=participantAccuracyColumns) 
        participantAccuracy = pd.concat([participantAccuracy, d])

        msk = np.random.rand(len(idf)) < 0.2
        train = idf[msk]
        
        x0 = [0.0]
        res = sp.optimize.minimize(Accuracy, x0, args=(model, train), method='L-BFGS-B', bounds=bnds, tol=1e-6)
        split_coef = res.x[0]

        test = idf[~msk]

        accuracy = 1 - (Accuracy(split_coef, test, True))
        
        d = pd.DataFrame([[id, coef, "UB-VAE", accuracy]], columns=participantAccuracyColumns) 
        participantAccuracy = pd.concat([participantAccuracy, d])

    participantAccuracy.to_pickle("splitAccuracy_UBVAE.pkl")
    sns.boxplot(data=participantAccuracy, x="Type", y="Accuracy")

    plt.show()


    