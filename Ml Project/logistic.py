import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


data = pd.read_csv('Blood_samples_dataset_balanced_2(f).csv')

data['Disease'].replace({'Anemia': 0, 'Healthy':1, 'Diabetes':2, 'Thalasse': 3, 'Thromboc': 4}, inplace=True)


x = data.iloc[:,0:24]
y = data.iloc[:,-1:]


x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size= 0.3, random_state= 34)
logss = LogisticRegression(random_state= 34,penalty= 'l2', C= 0.2)

logss.fit(x_train, y_train)
pred = logss.predict(x_test)

confs = metrics.confusion_matrix(y_test, pred)
print(confs)
dis = metrics.ConfusionMatrixDisplay(confusion_matrix= confs, display_labels=[0,1,2,3,4])
dis.plot()
# plt.show()

acc = metrics.accuracy_score(y_test,pred)
print(acc)
# print(logss.score(x_test,y_test))

names = ['Anemia', 'Healthy', 'Diabetes', 'Thalasse', 'Thromboc']
# print(classification_report(y_test, pred, target_names= names))