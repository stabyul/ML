import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

####  split the csv file into spam and ham
name = pd.read_csv('spambase_csv.csv')
x =name[name['class'] == 1]
y  =name[name['class'] == 0]

##### train ham and spam,  test ham and spam
train_spam = x.sample(n = 920)
test_spam = x.sample(n = 893)
train_ham = y.sample(n = 1380)
test_ham = y.sample(n = 1408)


####### concatenate train ham and spam into one dataframe
training = pd.concat([train_ham,train_spam], axis=0)
training1 = training.sample(frac = 1).reset_index(drop=True)

#### concatenate test ham and spam into one dataframe
testing = pd.concat([test_ham,test_spam], axis=0)
testing1 = testing.sample(frac = 1).reset_index(drop=True)

def preprocess(testing1):
    testing12 = testing1.iloc[:, 0:57]    ###### features
    testing1_y = testing1.iloc[:, -1:]      #### labels
    testing12_y = testing1_y.to_numpy()      ####### converting labels to numpy
    feature = testing12.to_numpy()          #### converting feature to numpy
    return testing12_y,feature

def prior_12(training1):
    zeros = training1[training1['class'] == 0]
    zeros2 = training1[training1['class'] == 1]
    prior = len(zeros)/len(training1)         ## class prior when class = 0
    prior2 = len(zeros2)/len(training1)       ## class prior when class = 1
    return prior,prior2

def mean_std(training1):
    means = training1.groupby('class').mean()   ## mean for each feature grouped by class
    means_0 = means.loc[0]         ### mean for each feature at class 0
    means_1 = means.loc[1]         ## mean for each feature at class 1
    std1 = training1.groupby('class').std()       ## standard deviation for each feature grouped by class
    std1 = std1.replace(0,0.0001)      ########## replace standard deviation that have 0 with 0.0001
    std1_0 = std1.loc[0]     ## standard deviation for each feature at class 0
    std1_1 = std1.loc[1]    ## standard deviation for each feature at class 1

    #########converting the mean and standard deviation for each class######
    means_00 = means_0.to_numpy()
    means_11 = means_1.to_numpy()
    std1_00 = std1_0.to_numpy()
    std1_11 = std1_1.to_numpy()
    return means_00, means_11, std1_00, std1_11




def test_i(feature, means_00, means_11, std1_00, std1_11, prior, prior2):
    correct = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0 
    correct5 = 0
    confusion = np.zeros((2,2))
    for i in range(feature.shape[0]):
        totals = []
        ###### p(x/c) for each class
        normal = 1 / (std1_00 * np.sqrt(2 * np.pi)) * np.exp(-((feature[i,:] - means_00))**2 / (2 * (std1_00 ** 2)))
        normal2 = 1 / (std1_11 * np.sqrt(2 * np.pi)) * np.exp(-((feature[i,:] - means_11))**2 / (2 * (std1_11 ** 2)))
        #### classification calculation for both class
        a = np.sum(np.log(normal))
        b = np.log(prior)
        total = a + b
        totals.append(total)   ## store classification for the first class in a list
        c = np.sum(np.log(normal2))
        d = np.log(prior2)
        total2 = c + d
        totals.append(total2)   ## store the classification for second class in the same list

        totalol = np.argmax(totals)


        if totalol == testing12_y[i]:        ## calculating accuracy
            correct += 1
    
        confusion[testing12_y[i,0]][totalol] += 1   ### confusion matrix
    
       ######### formula for calculating the metrics

        if totalol == 1 and testing12_y[i] == 1:
            correct2 += 1
        
        if totalol == 1 and testing12_y[i] == 0:
            correct3 += 1

        if totalol == 0 and testing12_y[i] == 1:
            correct4 += 1

        if totalol == 0 and testing12_y[i] == 0:
            correct5 += 1


    
    #### print metrics
    print('ACCURACY', correct/len(feature))
    print('SAME ACCURACY BUT DIFFERENT FORMULA:', (correct2 + correct5)/ (correct2 + correct3 + correct4 + correct5))
    print('PRECISION:', (correct2)/ (correct2 + correct3))
    print('RECALL:', (correct2) / (correct2 + correct4))  
    
    ### plotting for aconfusion matrix
    plt.figure(figsize=(10,10), dpi = 110)
    s = sns.heatmap(confusion,annot = True)
    s.set_xlabel("Predicted Label")
    s.set_ylabel("Actual Label")
    s.set_title('Confusion matrix')
    plt.show()



testing12_y,feature = preprocess(testing1)
prior,prior2 = prior_12(training1)
means_00, means_11, std1_00, std1_11 = mean_std(training1)
test_i(feature, means_00, means_11, std1_00, std1_11, prior, prior2)

