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