import pandas as pd
import numpy as np


# Load dataset files
dataset = pd.read_csv("spambase/spambase.data", header = None)

# Get basic information of dataset
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

result = {
    'Model': [],
    'Accuracy': []
}

# Evaluate the accuracy of SVM
def evaluate_SVM(pred_data, real_data, name_data):
    check_same = pred_data == real_data
    accuracy = sum(check_same) / len(check_same) * 100
    print(name_data, "Accuracy: ", accuracy, "%")
    # To match the scale for ploting
    return accuracy / 100

def present_SVM(SVM_Model):
    # Use the model to predict the training and test sets.
    train_data = SVM_Model.predict(X_train.values)
    test_data = SVM_Model.predict(X_test.values)

    # Evaluate the model using the training and test sets
    evaluate_SVM(train_data, y_train, 'Train')
    return evaluate_SVM(test_data, y_test, 'Test')

# Linear model for SVM
SVM_Model = svm.SVC(kernel = 'linear').fit(X_train, y_train)
svm_linear_accuracy = present_SVM(SVM_Model)
result['Model'].append("SVM_linear")
result["Accuracy"].append(svm_linear_accuracy)

# Sigmoid kernal for SVM
SVM_Model = svm.SVC(kernel = 'sigmoid').fit(X_train, y_train)
svm_sigmoid_accuracy = present_SVM(SVM_Model)
result['Model'].append("SVM_signmoid")
result["Accuracy"].append(svm_sigmoid_accuracy)

# Polynomial kernal for SVM
SVM_Model = svm.SVC(kernel = 'poly').fit(X_train, y_train)
svm_poly_accuracy = present_SVM(SVM_Model)
result['Model'].append("SVM_poly")
result["Accuracy"].append(svm_poly_accuracy)



from keras.utils import to_categorical

# Reformat outputs to categorical values
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build neural network model
import tensorflow as tf
import keras

model = keras.models.Sequential()

# Input layer & hidden layer
model.add(keras.layers.Dense(units=24, input_dim = 57, activation = 'relu'))

# Hidden layer
model.add(keras.layers.Dense(units=24, activation = 'relu'))

# Output layer
model.add(keras.layers.Dense(units=2, activation = tf.nn.softmax))

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# Fit the model
print('Starting training')

training_stats = model.fit(X_train, y_train, epochs = 10)

print('Training finished')

# Evaluation
evaluation = model.evaluate(X_test, y_test, verbose=0)

print('Test Set Evaluation: loss = %0.6f, accuracy = %0.2f%%' %(evaluation[0], 100 * evaluation[1]))

# Add evaluation information in result dictionary
result['Model'].append("Neural Network")
result["Accuracy"].append(evaluation[1])

# Print out result

data = pd.DataFrame.from_dict(result)
data.plot.bar(x = 'Model', y = 'Accuracy', rot = 20)

# Save the plot
import matplotlib.pyplot as plt

plt.savefig("result.pdf")

