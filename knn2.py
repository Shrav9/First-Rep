# First Code on Github
#Implementing k-NN Algorithm


#Loading libraries 

from sklearn import preprocessing , model_selection
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as np

#Loading Iris dataset  #sepal length, sepal width, petal length, petal width
path = 'C:\\Users\\dasari.shravani\\Desktop\\Python\\iris1.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv(path, names=names)


#Split train and test dataset
array = data.values
X = array[:,0:4]
Y = array[:,4]
#Y=Y.astype('int')
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.2)


print('X_Train :' +repr(len(X_train)))
print('Y_Train :' +repr(len(Y_train)))
print('X_Test  :' +repr(len(X_test)))
print('Y_Test  :' +repr(len(Y_test)))

#k-NN Algorithm
knn = KNeighborsClassifier(n_neighbors=1) 
knn.fit(X_train,Y_train) 

#Accuracy
accuracy = knn.score(X_test, Y_test)     #accuracy = matches/samples
 
#Testing with a sample 
x_new = np.array([[5.0,2.5,4.0,1.2]]) 
prediction = knn.predict(x_new) 
  
#print("Predicted target value: {} \n".format(prediction)) 
print("Predicted feature name: {} \n".format(prediction))
print("Accuracy: {:.2f}".format(accuracy))