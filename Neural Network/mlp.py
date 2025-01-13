import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import random

#training and testing data for mnist
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')


#prepocessing the data
def preprocess(value):
    X = value.iloc[:,1:785]
    X = X.to_numpy()
    X = X / 255
    # bias = np.ones((len(value),1))
    # X= np.concatenate((X,bias),axis=1)
    Y = value.iloc[:,0]
    Y = Y.to_numpy()
    print(X.shape)

    print(Y.shape)
    return X,Y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(output1):
     y_pred = np.argmax(output1)
     return y_pred

def accurate(train_x,w,bias,w2,bias2,neuron):
        correct = 0
        for n in range(len(train_x)):
            output = np.dot(train_x[n],w) + bias
            output = sigmoid(output)
            output = np.reshape(output,(1,neuron[0]))
            # print(output.shape)           
            
            for j in range (len(output)):
                output1 = np.dot(output[j],w2) + bias2
                output1 = sigmoid(output1)

            if train_y[n] == predict(output1):
                 correct +=1
        return correct / len(train_x)

def accurate_test(test_x,w,bias,w2,bias2,neuron):
        correct2 = 0
        for k in range(len(test_x)):
            output = np.dot(test_x[k],w) + bias
            output = sigmoid(output)
            output = np.reshape(output,(1,neuron[0]))
            # print(output.shape)           
            
            for m in range (len(output)):
                output1 = np.dot(output[m],w2) + bias2
                output1 = sigmoid(output1)

            if test_y[k] == predict(output1):
                 correct2 +=1
        return correct2 / len(test_x)

# def shuf (train_x,train_y):
#      train_x,train_y = shuffle(train_x, train_y)
#      return train_x, train_y

def acc_ep(train_x,w,bias,w2,bias2,neuron):
    correct3 = 0
    for h in range(len(train_x)):
        output = np.dot(train_x[h],w) + bias
        output = sigmoid(output)
        output = np.reshape(output,(1,neuron[0]))
        # print(output.shape)           
        
        for p in range (len(output)):
            output1 = np.dot(output[p],w2) + bias2
            output1 = sigmoid(output1)
        if train_y[h] == predict(output1):
                 correct3 +=1
    return correct3 / len(train_x)
    


def train_test(train_x,train_y,test_x,test_y, epoch,lr,alpha,neuron):
    w = np.random.uniform(-0.05,0.05, size=(784,neuron[0]))
    w2 = np.random.uniform(-0.05,0.05, size=(neuron[0],10))
    w_change = np.zeros((784,neuron[0]))
    w2_change = np.zeros((neuron[0],10))
    bias = np.zeros((1,neuron[0]))
    bias2 = np.zeros((1,10))
    acc_train = []
    acc2_test = []
    
    
    # train_temp = train_x.copy()
    # train2_temp = train_y.copy()
    # data = np.arange(train_x.shape[0])
    for h in range(len(train_x)):
        output = np.dot(train_x[h],w) + bias
        output = sigmoid(output)
        output = np.reshape(output,(1,neuron[0]))
        # print(output.shape)           
        
        for p in range (len(output)):
            output1 = np.dot(output[p],w2) + bias2
            output1 = sigmoid(output1)

    acc3 = acc_ep(train_x,w,bias,w2,bias2,neuron)
    acc_train.append(acc3)
    print("accuracy for epoch 0 is", acc3)

    # shuf = np.random.permutation(len(train_x))
    # train_x = train_x[shuf]  
    # train_y = train_y[shuf]
   
    for i in range(epoch):
        A = train_x
        B = train_y
        A ,B = shuffle(A, B)
        for n in range(len(train_x)):
            output = np.dot(train_x[n],w) + bias
            output = sigmoid(output)
            output = np.reshape(output,(1,neuron[0]))
            # print(output.shape)           
            
           
            # bias2 = np.ones((1,1))
            # output = np.concatenate((output,bias2), axis = 1)
            for j in range (len(output)):
                output1 = np.dot(output[j],w2) + bias2
                output1 = sigmoid(output1)
                # print(output1.shape)

    
            t = np.full(10, 0.1)
            t[train_y[n]] = 0.9
            # print(t)
        

            #error 
            error_output = output1 * (1 - output1) * (t - output1)
            # print(error_output)
            error_hidden = output * (1 - output) * np.dot(error_output,np.reshape(w2,(10,neuron[0])))
            # print(error_hidden)
           
            # # weight update

            w2_change = lr * np.dot(np.reshape(output,(neuron[0],1)), error_output) + alpha * w2_change
            w2 += w2_change
            # print(w2)
            
            w_change = lr * np.dot(np.reshape(train_x[n],(784,1)),error_hidden) + alpha * w_change
            w = w + w_change
        acc = accurate(train_x,w,bias,w2,bias2,neuron)
        acc_train.append(acc)
        print("accuracy for epoch", i + 1, "is", acc)

        for k in range(len(test_x)):
            output = np.dot(test_x[k],w) + bias
            output = sigmoid(output)
            output = np.reshape(output,(1,neuron[0]))
            # print(output.shape)           
            
            for m in range (len(output)):
                output1 = np.dot(output[m],w2) + bias2
                output1 = sigmoid(output1) 

        # acc = accurate(train_x,w,bias,w2,bias2)
        # print("accuracy for epoch", i + 1, "is", acc)

        acc2 = accurate_test(test_x,w,bias,w2,bias2,neuron)
        acc2_test.append(acc2)
        print("accuracy for epoch", i + 1, "is", acc2)



    # confusion = np.zeros((10,10))
    # for k in range(len(test_x)):
    #     output = np.dot(test_x[k],w) + bias
    #     output = sigmoid(output)
    #     output = np.reshape(output,(1,20))
    #     # print(output.shape)           
        
    #     for m in range (len(output)):
    #         output1 = np.dot(output[m],w2) + bias2
    #         output1 = sigmoid(output1)

    #     y_pred = predict(output1)
    #     confusion[test_y[k]][y_pred] += 1

    # plt.figure(figsize=(10,10), dpi = 110)
    # s = sns.heatmap(confusion,annot = True)
    # s.set_xlabel("Predicted Label")
    # s.set_ylabel("Actual Label")
    # s.set_title('Confusion matrix using learning rate = 0.1')

    # plt.show()
                
    return acc_train,acc2_test
 
   

train_x, train_y = preprocess(train_data)
test_x, test_y = preprocess(test_data)
# train_x, train_y =  shuf(train_x, train_y)
acc_train,acc2_test = train_test(train_x, train_y,test_x,test_y,epoch = 50,lr = 0.1, alpha = 0.9, neuron = [20,50,100])

# plotting
plt.plot(acc_train, color = 'red', label = 'train')
plt.plot(acc2_test, color = 'green', label = 'test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs using learning rate = 0.1')
plt.legend()
plt.show()
