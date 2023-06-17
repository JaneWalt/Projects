import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Importing the dataset
Iris = pd.read_csv('C:/Users/User/Desktop/Iris dataset.csv')
print(Iris.info())
print(Iris.head())
print(Iris.describe())

# Separating features and labels
x = Iris.iloc[:,:4] #features
y = Iris.iloc[:,-1]  #labels
print(x.head())
print(y.head())

# Splitting the dataset
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.30,random_state=True)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Training the model
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
knn.fit(X_train, y_train)

#Prediction
print(X_test)
Y_pred = knn.predict(X_test)
Y_pred2 = knn.predict([[5.1,3.5,1.4,0.2]])
print(Y_pred)
print(Y_pred2)
print("Accuracy:",accuracy_score(y_test, Y_pred))
