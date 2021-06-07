import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

default = pd.read_csv("default.csv")

print(default.head())

print(default.isnull().sum())

print(default.shape)

# Analyze the student !
print(default['student'].value_counts())

# Analyze the default !
print(default['default'].value_counts())

# Analyze the income 
print(default['income'].value_counts())

# Analyze the balance 
print(default['balance'].value_counts())

# make a bar plot !
plt.figure(figsize=(8,5))
plt.bar(list(("No","Yes")),list(default['student'].value_counts()),color=['blue','yellow'])
plt.title("Student Analysis")
plt.show()

# make a bar plot default
plt.bar(list(("No","Yes")),list(default['default'].value_counts()),color=['olive','green'])
plt.title('Default Analysis')
plt.show()

# Student Vs Default
plt.scatter(x='student',y='default',data=default)
plt.title("Student Vs Default")
plt.show()

# Balance Analysis !
plt.figure(figsize=(8,5))
plt.scatter(x='balance',y='income',data=default,color='pink')
plt.show()

# Split the model into train & test

x = default[['balance']]
y = default[['default']]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)

print(x_train.head())

print(x_test.head())

print(y_train.head())

print(y_test.head())

# Implemented the support vector machine

from sklearn.svm import SVC

svc = SVC()

print(svc.fit(x_train,y_train))

y_pred = svc.predict(x_test)

# Actual Values !
print(y_test.head())

# Predicted Values
print(y_pred[0:5])

# make a confusion matrix to check the accuracy !

from sklearn.metrics import confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))

# Check the accuracy of ml model
print(accuracy_score(y_test,y_pred))



