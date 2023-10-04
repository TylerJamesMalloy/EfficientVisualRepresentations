import numpy as np 

stim0 = np.load("./stimuli0.npy")
stim1 = np.load("./stimuli1.npy")
stim2 = np.load("./stimuli2.npy")
stim3 = np.load("./stimuli3.npy")
stim4 = np.load("./stimuli4.npy")
stim5 = np.load("./stimuli5.npy")

stims = np.concatenate((stim0, stim1, stim2, stim3, stim4, stim5))

print(stims.shape)

np.save("./stimuli.npy", stims)