#importing libararies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##importing dataset
dataset = pd.read_csv(r"C:\Users\15vik\OneDrive\Desktop\naresh it\extract\28th,29th\TASK-24\framingham.csv")
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values

#Taking care of missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x)
x = imputer.transform(x)


#Splitting Dataset

from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test=train_test_split(x, y, test_size=0.20, random_state=0)

#Feature Scaling for Improving model Performance

from sklearn.preprocessing import StandardScaler
sc_x =StandardScaler()
x_train =sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#Training the logistic regression model

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

#Predicting the test results
y_pred = classifier.predict(x_test)

#Evaluating Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Accuracy of model
from sklearn.metrics import accuracy_score
print('Accuracy of my model on testing set :' , accuracy_score(y_test, y_pred))


