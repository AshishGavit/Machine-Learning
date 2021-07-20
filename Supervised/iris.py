# import libraries
from sklearn.datasets import load_iris # iris dataset
from sklearn.model_selection import train_test_split # split data into two part training and testing
from sklearn.neighbors import KNeighborsClassifier # algorithm of machine learning - supervised - both classification and regression
import numpy as np # Numeric Python

iris_dataset = load_iris() # load the dataset into a variable
print("Target names: {}".format(iris_dataset['target_names'])) # get the target_names
print("Feature names: {}".format(iris_dataset['feature_names']))# get the feature names of iris dataset
print("Type of data: {}".format(type(iris_dataset['data']))) # use type() method to display the data type

print("Shape of data: {}".format(iris_dataset['data'].shape)) # display the shape of 'data'

# get 'target' type(), shape, and display 'target' colume
print("Type of target: {}".format(type(iris_dataset['target']))) 
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))

# split the data into tow parts 'training' and 'testing' using 'train_test_split()' method
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# print the shape of 'X_train' and 'y_train'
print("X_train shape: {}".format(X_train.shape)) # 'X_train' is the independent variable
print("y_train shape: {}".format(y_train.shape)) # 'y_train' is the dependent variable(target)

# print the shape of 'X_test' and 'y_test'
# this will be used to check if our model is able to predict correctly or not
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

knn = KNeighborsClassifier(n_neighbors=1) # n_neighbors parameter is used to define how many neighors to you want
knn.fit(X_train, y_train) # in each row in dataset fit 1 neighbors

X_new = np.array([[5, 2.9, 1, 0.2]]) # dummy value for predicting new value
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new) # predict() send the new value to the model we created by using KNeighorsClassifier()
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# X_new is a dummpy dataset, lets predict using the testing set
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score (np.mean): {:.2f}".format(np.mean(y_pred == y_test)))# check if the predicted value(y_pred) is equal to the traget value(y_test)

# We need confirmation weather our model is able to predict correctly if we give a new value to it
# How do we confirm?
# in the above we did 'np.mean(predect == target), but there is another Measure called 'score'
print("Test set score (knn.score): {:.2f}".format(knn.score(X_test, y_test)))

# step1: load the dataset
# step2: extract features and target variables
# step3: split the dataset to 'training' and 'testing' sets
# step4: use machine learning algorithm, in this case 'KNeighborsClassifier' (model = knn) Supervised- classification
# step5: predict the model(knn) using 'X_test'
# step6: Measure the accuracy using 'np.mean(predicted == target)' or 'score(X_test, y_test)