import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_train = pd.read_csv('mnist_train.csv')
data_test = pd.read_csv('mnist_test.csv')
# It has 10 digits ranging from 0-9
# x1 = data_train.iloc[:,1:785]   #data
# y1 = data_train.iloc[:,0]   #label


def preproces(data):
    input = np.array(data.iloc[:,1:785])
    input = input.astype(float)
    labels = np.array(data.iloc[:,0])
    input= input/225.0 #scaling
    input = np.concatenate((input,np.ones((len(data),1))),axis=1)
    print(input.shape)  #addition of the bias
    return input, labels

# def train(tData):
#    w = np.random.uniform(-0.05,0.05,(785,10))
#    epoch = 70
#    m = len(tData) 
#    alpha = 0.05
#    corect2 = 0
#    acc_train = []
#    acc_test = []
#   #  for l in range(m):
#   #    y1 = np.dot(input[l],w)
#   #    pred1 = np.argmax(y1)   #prediction
#   #    if pred1 == tTarget[l]:
#   #      corect2 += 1
#   #  acc_train.append(corect2/m)
#   #  print("Accuracy for epoch 0 is",corect2/m)
  
#   # train model on training data
#    for i in range(epoch):
#     corect = 0
#     corect_test = 0
#     for j in range(m):
#      y = np.dot(tData[j],w)
#      print(y.shape)
    #  pred = np.argmax(y)   #prediction
    #  if pred == tTarget[j]:
    #    corect += 1
    #  else:
    #    new_y = np.where(y>0,1,0)
    #    t = np.zeros(10)
    #    t[labels[j]] = 1
    #    w += alpha * np.dot(np.reshape(input[j],(785,1)), np.reshape((t-new_y),(1,10)))
    # # print accuracy of model on traning data for current epoch
    # acc_train.append(corect/m)
    # print("Accuracy for epoch",i+1," is",corect/m)
    
    # Test model on testing data and get accuracy
  #   for k in range(len(testdata)):
  #    y = np.dot(testdata[k],w)
  #    pred = np.argmax(y)   #prediction
  #    if pred == testtarget[k]:
  #      corect_test += 1
  #   acc_test.append(corect_test/len(testdata))
  #   print("Accuracy for epoch",i+1," is",corect_test/len(testdata))
   
  #  #confusion matrix
  #  confs = np.zeros((10,10))
  #  for k in range(len(testdata)):
  #    y = np.dot(testdata[k],w)
  #    pred = np.argmax(y)   #prediction
  #    confs[testtarget[k]][pred] += 1
  #  print(confs)
    


  #  return acc_train, acc_test


    
# preprocess traing data
tData, tTarget = preproces(data_train.iloc[0:48000, :])

# testdata, testtarget =preproces(data_test)
# acc_train, acc_test =train(tData,tTarget, testdata,testtarget)

# plt.plot(acc_train)
# plt.plot(acc_test)
# plt.show()
