import os 

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 


# 0 -> 2, 1 -> 3, 2-> 4
colors = pd.read_csv("./source/colors.csv")
color_numbers = []

for row, color in colors.iterrows():
    colors = color["colors"]
    colors = colors[1:-1]
    colors = colors.split(", ")
    colors = np.array(colors, dtype=int) 

    colors = [4 if x==0 else x for x in colors]
    colors = [3 if x==1 else x for x in colors]
    #colors = [2 if x==2 else x for x in colors] # redundant

    color_numbers.append(colors)



# "rt","key_press","reward","type","marble","trial_num","stim_1","stim_2","change_index","block","changed","marble_set"

dataColumns = ["Id", "Experiment Type", "Marble Set", "Block", "Trial", "Trial Type", "Reaction Time", "Left Stimulus", "Right Stimulus", "Left Stimulus Marbles", "Right Stimulus Marbles", "New Stimulus", "New Stimulus Marbles", "Changed", "Changed Index", "Key Pressed", "Reward", "Correct"]
dataFrame = pd.DataFrame([], columns=dataColumns)

Intcols = ["Marble Set", "Block", "Trial", "Trial Type", "Left Stimulus", "Right Stimulus", "Changed", "Correct"]
dataFrame[Intcols] = dataFrame[Intcols].applymap(np.int64)

learning_ids = [1049540426 ,
106381449 ,
1066486519 ,
1092721686 ,
1158286381 ,
1204911692 ,
1224801952 ,
1231600554 ,
125257521 ,
1271988871 ,
1361833784 ,
1424208338 ,
1472936997 ,
1480381967 ,
1599165604 ,
1612337210 ,
1683383581 ,
1789205864 ,
1804789041 ,
2068222707 ,
2160484470 ,
219334677 ,
2253047146 ,
2269165132 ,
2452276079 ,
2473448161 ,
248473192 ,
2485825004 ,
2551137904 ,
2572158007 ,
2608388359 ,
2731003481 ,
2734423660 ,
285880003 ,
2881640105 ,
2891310284 ,
2913156997 ,
2918206239 ,
2975869345 ,
3010113900 ,
3045818450 ,
3060944969 ,
3113381684 ,
3149734194 ,
3169950911 ,
317608556 ,
3365603469 ,
348300766 ,
3549067443 ,
3576860780 ,
3633378519 ,
3637610314 ,
3693949122 ,
3758324478 ,
376286187 ,
3865498490 ,
390755978 ,
4074885344 ,
4081939425 ,
4132587080 ,
4154479176 ,
4258732026 ,
440344663 ,
4506016898 ,
4513281267 ,
4573741990 ,
4647530528 ,
4715121391 ,
4753076799 ,
4758191482 ,
4773591768 ,
4796991397 ,
4799514765 ,
481425382 ,
4819188505 ,
4858293512 ,
4919091392 ,
4934510627 ,
4971494324 ,
501044799 ,
5027782038 ,
5150448125 ,
5176739543 ,
5265534006 ,
5336978760 ,
5412622743 ,
5522004035 ,
5559758084 ,
5580217541 ,
5673514813 ,
5892509075 ,
5904028522 ,
5906483058 ,
6074039749 ,
6142644684 ,
614690450 ,
6190914712 ,
6314725237 ,
6499217974 ,
6506762788 ,
6652616958 ,
6690243889 ,
6764555397 ,
6899704486 ,
6945026478 ,
7013814928 ,
7106456477 ,
7125339922 ,
7178847280 ,
7198253621 ,
7211046746 ,
728983901 ,
7291120861 ,
7509475451 ,
7633668871 ,
7711000091 ,
7746857865 ,
7782241223 ,
7835923721 ,
7840412677 ,
7863930389 ,
7869458961 ,
7968022668 ,
7969693716 ,
7972392719 ,
8070693962 ,
8133224429 ,
8181295670 ,
8483879839 ,
8485002306 ,
8499283501 ,
8557939177 ,
8759020784 ,
8776113055 ,
8813743174 ,
8915475951 ,
9057819681 ,
9084246693 ,
9137224319 ,
9152362149 ,
9225231886 ,
9273892904 ,
9312196920 ,
9329966902 ,
9348576762 ,
9547662512 ,
9557380883 ,
9748880425 ,
978150698 ,
9840397114 ]

for filenum, filename in enumerate(os.listdir("./data/")):
    filedata = pd.read_csv("./data/" + filename)
    filedata = filedata.dropna()

    if(len(filedata.columns) != 13): continue 
    if(len(filedata["marble_set"].unique()) != 1): continue
    id = filename.split("_")[0]
    marble_set = filedata["marble_set"].unique()[0]

    if(len(filedata) < 200): continue

    filedata = filedata.tail(200)
    first_trial = int(filedata["trial_num"].to_list()[0])

    for row, data in filedata.iterrows():
        block = data["block"]
        trial = data["trial_num"]
        trial = int((trial - first_trial) - (20 * block))
        trial_type = "Change Detection" if data["type"] == 0 else "Utility Selection"
        reaction_time = data["rt"]
        left_stimulus = int(data["stim_1"])
        right_stimulus = int(data["stim_2"])
        new_stimulus = int(data["new_stim"])
        left_stimulus_marbles = color_numbers[int(data["stim_1"])]
        right_stimulus_marbles  = color_numbers[int(data["stim_2"])]
        new_stimulus_marbles = color_numbers[int(data["new_stim"])]
        changed = int(data["changed"])
        changed_index = int(data["change_index"])
        key_press = data["key_press"]
        reward = data["reward"]

        left_utility_mean = np.mean(color_numbers[int(data["stim_1"])])
        right_utility_mean = np.mean(color_numbers[int(data["stim_2"])])

        experiment_type = "Learning" if int(id) in learning_ids else "Decision-Making"

        correct = 0
        if(trial_type == "Change Detection"): # j or k, testing changed 
            if((changed == 1 and key_press == 'k') or (changed == 0 and key_press == 'j')):
                correct = 1
        else: # d or f, testing utilty selection
            if((key_press == "d" and left_utility_mean >= right_utility_mean) or (key_press == "f" and left_utility_mean <=right_utility_mean)):
                correct = 1 

        d = pd.DataFrame([[id, experiment_type, marble_set, block, trial, trial_type,  reaction_time, left_stimulus, right_stimulus, left_stimulus_marbles, right_stimulus_marbles, new_stimulus, new_stimulus_marbles, changed, changed_index, key_press, reward, correct]], columns=dataColumns)
        dataFrame = pd.concat([dataFrame, d], ignore_index=True)




print(dataFrame)
dataFrame.to_csv("./clean/participantData.csv")