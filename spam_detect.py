import pandas as pd
import numpy as np


# loading dataset files
dataset = pd.read_csv("spambase/spambase.data", header = None)

# getting basic information of dataset
print(dataset.head())
print(dataset.shape)