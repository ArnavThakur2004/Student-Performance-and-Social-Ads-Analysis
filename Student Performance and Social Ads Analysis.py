# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:49:31 2024

@author: arnav
"""



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score



df= pd.read_csv("Student_Marks.csv")


plt.scatter(df['Hours'], df['Scores'])
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

x = df[['Hours']]
y = df['Scores']

print("X model",x)
print(" Y model",y)


X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=2)
lr = LinearRegression()
lr.fit(X_train, Y_train)
   
prediction = lr.predict(X_test.iloc[2].values.reshape(1,1))
print("\nPrediction of data using Linear Regression:\n",prediction)

plt.scatter(df['Hours'],df['Scores'])
plt.plot(X_test,lr.predict(X_test),color='yellow')
plt.xlabel("Hours")
plt.ylabel("Scores")
##plt.show()
Y_pred = lr.predict(X_test)
r2 = r2_score(Y_test,Y_pred)
print("r2 score of the linear ",r2)
dataset =pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression:", accuracy)
