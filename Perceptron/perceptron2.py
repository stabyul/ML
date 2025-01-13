import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#training and testing data for mnist
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')


#prepocessing the data
def preprocess(value):
    X = value.iloc[:,1:785]
    X = X.to_numpy()
    X = X / 255
    bias = np.ones((len(value),1))
    X= np.concatenate((X,bias),axis=1)
    Y = value.iloc[:,0]
    Y = Y.to_numpy()
    return X,Y


def train_test(x_train,y_train, x_test, y_test):
    epoch = 70
    lr = 0.1
    w = np.random.uniform(-0.05,0.05, size=(785,10))
    acc_train = []
    acc_test = []
    correct1 = 0
    #before training epoch
    for n in range(len(x_train)):
        output = np.dot(x_train[n],w)
        y_pred = np.argmax(output)
        if y_pred == y_train[n]:
            correct1 += 1
    acc_train.append(correct1 / len(x_train))
    print("accuracy for epoch 0 is", correct1 / len(x_train))
    for i in range(epoch):
      Correct = 0
      correct2 = 0
      for j in range(len(x_train)):
        output = np.dot(x_train[j],w)
        y_pred = np.argmax(output)
        if y_pred == y_train[j]:
            Correct += 1
        else:
            y_condition = np.where(output>0,1,0)
            t = np.zeros(10)
            t[y_train[j]] = 1
            w += lr * np.dot(np.reshape(x_train[j], (785,1)),np.reshape(t,(1,10))- y_condition)
      acc_train.append(Correct / len(x_train))
      print("accuracy for epoch", i + 1, "is", Correct / len(x_train))
#testing
      for k in range(len(x_test)):
        output = np.dot(x_test[k],w)
        y_pred = np.argmax(output)
        if y_pred == y_test[k]:
            correct2 += 1
      acc_test.append(correct2 / len(x_test))
      print("accuracy for epoch", i + 1, "is", correct2 / len(x_test))

    confusion = np.zeros((10,10))
    for k in range(len(x_test)):
        output = np.dot(x_test[k],w)
        y_pred = np.argmax(output)
        confusion[y_test[k]][y_pred] += 1
    # print(confusion)

    plt.figure(figsize=(10,10), dpi = 110)
    s = sns.heatmap(confusion,annot = True)
    s.set_xlabel("Predicted Label")
    s.set_ylabel("Actual Label")
    s.set_title('Confusion matrix using learning rate = 0.1')

    plt.show()

    return acc_train, acc_test



x_train,y_train = preprocess(train_data)
x_test, y_test = preprocess(test_data)
acc_train, acc_test = train_test(x_train,y_train, x_test, y_test)

# plotting
plt.plot(acc_train, color = 'red', label = 'train')
plt.plot(acc_test, color = 'green', label = 'test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs using learning rate = 0.1')
plt.legend()
plt.show()

