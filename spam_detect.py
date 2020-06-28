import pandas as pd
import numpy as np


# loading dataset files
dataset = pd.read_csv("spambase/spambase.data", header = None)

# getting basic information of dataset
print(dataset.head())
print(dataset.shape)

# Check if there are any empty values
print(pd.isnull("dataset"))
# The output is false which means there is no empty value in the dataset

# Use the train_test_split function to make a split of 20% data for testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(dataset.columns[57], axis=1), dataset[57],
                                                    test_size=0.2, random_state=42)

# Reset the index in the dataset
X_train = X_train.reset_index(drop = True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop=True)

# As the scale in the column 54, 55 and 56 are different from others, so I need to change these to the same scale of [0,100]
def Rescale_data(data):
    return ((data - np.min(data)) / np.max(data))
for a in [54,55,56]:
    X_train[a] = 100 * Rescale_data(X_train[a])
    X_test[a] = 100 * Rescale_data((X_test[a]))

# Check the scale
print(X_train.min(), X_train.max())
print(X_test.min(), X_test.max())
# After checking, all data are in the range of [0,100]

from sklearn import svm

# Evaluate the accuracy of SVM
def evaluate_SVM(pred_data, real_data, name_data):
    check_same = pred_data == real_data
    accuracy = sum(check_same) / len(check_same) * 100
    print(name_data, "Accuracy: ", accuracy, "%")

def present_SVM(SVM_Model):
    # Use the model to predict the training and test sets.
    train_data = SVM_Model.predict(X_train.values)
    test_data = SVM_Model.predict(X_test.values)

    # Evaluate the model using the training and test sets
    evaluate_SVM(train_data, y_train, 'Train')
    evaluate_SVM(test_data, y_test, 'Test')

# Linear model for SVM
SVM_Model = svm.SVC(kernel = 'linear').fit(X_train, y_train)
present_SVM(SVM_Model)

# Sigmoid kernal for SVM
SVM_Model = svm.SVC(kernel = 'sigmoid').fit(X_train, y_train)
present_SVM(SVM_Model)

# Polynomial kernal for SVM
SVM_Model = svm.SVC(kernel = 'poly').fit(X_train, y_train)
present_SVM(SVM_Model)
