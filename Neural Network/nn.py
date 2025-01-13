import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

#training and testing data for mnist
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

################preprocessing function##################

def preprocess(value):    
    X = value.iloc[:,1:785]
    X = X.to_numpy()
    X = X / 255
    Y = value.iloc[:,0]
    Y = Y.to_numpy()
    return X,Y

########function for calculating sigmoid#########
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

###############function to predict output of the output layer##########

def predict(output1):
    y_pred = np.argmax(output1)
    return y_pred

###############function to computer accuracy for train data######################

def accurate(train_x,w,bias,w2,bias2,neuron):
    correct = 0
    for n in range(len(train_x)):
        output = np.dot(train_x[n],w) + bias
        output = sigmoid(output)
        output = np.reshape(output,(1,neuron[2]))

        for j in range(len(output)):
            output1 = np.dot(output[j],w2) + bias2
            output1 = sigmoid(output1)

        if train_y[n] == predict(output1):
            correct += 1
    return correct / len(train_x)

##################function to compute accuracy for test data#####################

def accurate_test(test_x,w,bias,w2,bias2,neuron):
    correct2 = 0
    for k in range(len(test_x)):
        output = np.dot(test_x[k],w) + bias
        output = sigmoid(output)
        output = np.reshape(output,(1,neuron[2]))

        for m in range(len(output)):
            output1 = np.dot(output[m],w2) + bias2
            output1 = sigmoid(output1)

        if test_y[k] == predict(output1):
            correct2 += 1
    return correct2 / len(test_x)

######################### function for accuracy for epoch 0. it was not used for this assignment#############################

# def acc_ep0(train_x,w,bias,w2,bias2,neuron):
#     correct3 = 0
#     for h in range(len(train_x)):
#         output = np.dot(train_x[h],w) + bias
#         output = sigmoid(output)
#         output = np.reshape(output,(1,neuron[0]))

#         for p in range(len(output)):
#             output1 = np.dot(output[p],w2) + bias2
#             output1 = sigmoid(output1)

#         if train_y[h] == predict(output1):
#             correct3 += 1
#     return correct3 / len(train_x)



########################training function##############################
def train_test(train_x,train_y,test_x, lr, epoch,alpha, neuron):
    w = np.random.uniform(-0.05,0.05, size=(784,neuron[2]))  #weight for input to hidden layer
    w2 = np.random.uniform(-0.05,0.05, size=(neuron[2],10))  # weight for hidden to output layer
    bias = np.zeros((1,neuron[2]))    # bias for input layer
    bias2 = np.zeros((1,10))    #bias for hidden layer
    w_change = np.zeros((784,neuron[2]))  # change in weight for input to hidden layer
    w2_change = np.zeros((neuron[2],10))     #change in weight for hidden to output layer
    acc_train = []   #store train accuracy
    acc2_test = []    # store test accuracy


    ###################### same accuracy for epoch 0 ##########################################

    # for h in range(len(train_x)):
    #     output = np.dot(train_x[h],w) + bias
    #     output = sigmoid(output)
    #     output = np.reshape(output,(1,neuron[0]))

    #     for p in range(len(output)):
    #         output1 = np.dot(output[p],w2) + bias2
    #         output1 = sigmoid(output1)
    # acc3 = acc_ep0(train_x,w,bias,w2,bias2,neuron)
    # acc_train.append(acc3)
    # print("accuracy for epoch 0 is", acc3)


    for i in range(epoch):
        A = train_x
        B = train_y
        A, B = shuffle(A,B)   #shuffling of the data
        for n in range(len(train_x)):          # input to hidden layer
            output = np.dot(train_x[n],w) + bias
            output = sigmoid(output)
            output = np.reshape(output,(1,neuron[2]))

            for j in range(len(output)):        #hidden to output layer
                output1 = np.dot(output[j],w2) + bias2
                output1 = sigmoid(output1)

            t = np.full(10, 0.1)   # predicted labels
            t[train_y[n]] = 0.9

################calculating the errors #######################
            error_output = output1 * (1 - output1) * (t - output1)

            error_hidden = output * (1 - output) * np.dot(error_output,np.reshape(w2,(10,neuron[2])))

##########################performing weight update ######################
            w2_change = lr * np.dot(np.reshape(output,(neuron[2],1)), error_output) + alpha[0] * w2_change
            w2 += w2_change
            # print(w2)
            
            w_change = lr * np.dot(np.reshape(train_x[n],(784,1)),error_hidden) + alpha[0] * w_change
            w = w + w_change

        acc = accurate(train_x,w,bias,w2,bias2,neuron)
        acc_train.append(acc)
        print("accuracy for epoch", i + 1, "is", acc)   # accuracy of train data

##################### implementing the test data after training#################
        for k in range(len(test_x)):
            output = np.dot(test_x[k],w) + bias
            output = sigmoid(output)
            output = np.reshape(output,(1,neuron[2]))

            for m in range(len(output)):
                output1 = np.dot(output[m],w2) + bias2
                output1 = sigmoid(output1)


        acc2 = accurate_test(test_x,w,bias,w2,bias2,neuron)
        acc2_test.append(acc2)
        print("accuracy for epoch", i + 1, "is", acc2)   # accuracy of test data


############################confusion matrix of the test data###############################
    confusion = np.zeros((10,10))
    for k in range(len(test_x)):
        output = np.dot(test_x[k],w) + bias
        output = sigmoid(output)
        output = np.reshape(output,(1,neuron[2]))

        for m in range(len(output)):
            output1 = np.dot(output[m],w2) + bias2
            output1 = sigmoid(output1)

        y_pred = predict(output1)
        confusion[test_y[k]][y_pred] += 1

    plt.figure(figsize=(10,10), dpi = 110)
    s = sns.heatmap(confusion,annot = True)
    s.set_xlabel("Predicted Label")
    s.set_ylabel("Actual Label")
    s.set_title('Confusion matrix for one half of dataset using 100 neurons with alpha/momentum = 0.9')

    plt.show()

    return acc_train,acc2_test


########################## comment out the experiment 3 when running for the fist and second experiment#######
train_x, train_y = preprocess(train_data)        
train_x, train_y = preprocess(train_data.iloc[0:15000, :])   #used for experiment 3
train_x, train_y = preprocess(train_data.iloc[0:30000, :])    # used for experiment 3
test_x, test_y = preprocess(test_data)
acc_train, acc2_test = train_test(train_x,train_y,test_x, lr = 0.1, epoch = 50, alpha = [0.9, 0, 0.25, 0.5] ,neuron = [20,50,100])

#####the last line above shows different alphas or momentum and neurons used. 
####the index should be changed in the data in order to access them ######

# plotting graphs
plt.plot(acc_train, color = 'red', label = 'train')
plt.plot(acc2_test, color = 'green', label = 'test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs for one half of dataset using 100 neurons with alpha/momentum = 0.9')
plt.legend()
plt.show()

