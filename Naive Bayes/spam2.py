import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


name = pd.read_csv('spambase_csv.csv')
x =name[name['class'] == 1]
y  =name[name['class'] == 0]
# print(x)
# print(y)
train_spam = x.sample(n = 920)
test_spam = x.sample(n = 893)
train_ham = y.sample(n = 1380)
test_ham = y.sample(n = 1408)


# print(train_spam , test_spam , train_ham , test_ham)

training = pd.concat([train_ham,train_spam], axis=0)
training1 = training.sample(frac = 1).reset_index(drop=True)
testing = pd.concat([test_ham,test_spam], axis=0)
testing1 = testing.sample(frac = 1).reset_index(drop=True)
testing12 = testing1.iloc[:, 0:57]
testing1_y = testing1.iloc[:, -1:]
testing12_y = testing1_y.to_numpy()
# print(testing12_y.shape[0])
feature = testing12.to_numpy()
# print(feature)

zeros = training1[training1['class'] == 0]
zeros2 = training1[training1['class'] == 1]
prior = len(zeros)/len(training1)
prior2 = len(zeros2)/len(training1)
# prior4 = np.atleast_1d(prior)
# print(prior4.shape)
# print(prior2)


means = training1.groupby('class').mean()
means_0 = means.loc[0]
means_1 = means.loc[1]
std1 = training1.groupby('class').std()
std1 = std1.replace(0,0.0001)
std1_0 = std1.loc[0]
std1_1 = std1.loc[1]
# print(std1_0)

means_00 = means_0.to_numpy()
means_11 = means_1.to_numpy()
std1_00 = std1_0.to_numpy()
std1_11 = std1_1.to_numpy()
# print(feature.shape[1])
correct = 0
correct2 = 0
correct3 = 0
correct4 = 0 
correct5 = 0
confusion = np.zeros((2,2))
for i in range(testing12_y.shape[0]):
    totals = []
    normal = 1 /(std1_00 * np.sqrt(2 * np.pi)) * np.exp(-((feature[i,:] - means_00))**2/ (2 * (std1_00 **2)))
    normal2 = 1 /(std1_11 * np.sqrt(2 * np.pi)) * np.exp(-((feature[i,:] - means_11)**2)/ (2 * (std1_11 **2)))
    # print(normal.shape)
    a = np.sum(np.log(normal))
    b = np.log(prior)
    total = a + b
    totals.append(total)
    c = np.sum(np.log(normal2))
    d = np.log(prior2)
    total2 = c + d
    totals.append(total2)

    totalol = np.argmax(totals)
    # print(totalol)

    if totalol == testing12_y[i]:
        correct += 1
   
    confusion[testing12_y[i,0]][totalol] += 1
    # print(confusion)


    if totalol == 1 and testing12_y[i] == 1:
        correct2 += 1
    
    if totalol == 1 and testing12_y[i] == 0:
        correct3 += 1

    if totalol == 0 and testing12_y[i] == 1:
        correct4 += 1

    if totalol == 0 and testing12_y[i] == 0:
        correct5 += 1




# print('total', correct/len(testing12_y))
print(correct2)
print(correct3)
print(correct4)
print(correct5)

print('total', correct/len(testing12_y))
print('accuracy', (correct2 + correct5)/ (correct2 + correct3 + correct4 + correct5))
print('percision:', (correct2)/ (correct2 + correct3))
print('recall', (correct2) / (correct2 + correct4))  

plt.figure(figsize=(10,10), dpi = 110)
s = sns.heatmap(confusion,annot = True)
s.set_xlabel("Predicted Label")
s.set_ylabel("Actual Label")
s.set_title('Confusion matrix using learning rate = 0.1')
plt.show()






