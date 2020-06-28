# MSA2020 - AI & Advanced Analytics (Spam emails detection)

## Table of Contents
1. [Overview](#Overview)
2. [Features](#Features)
3. [Dependencies](#Dependencies)
4. [Running the project](#Runningtheproject)
5. [Step by step instructions](#Stepbystepinstructions)
6. [Results](#Results)

### 1. Overview
In recent years, spam emails are increasing dramatically, so how to identify spam emails is crucial for programmers to solve, so I made this project by applying several different models.

### 2. Features
* 4 machine learning models
* Present plot of accuracy for different models

### 3. Dependencies
* python (3.8)
* pandas
* numpy
* sklearn
* keras
* tensorflow
* matplotlib

### 4. Running the project
* Install all dependencies (if needed)
* Check the local machine requirement to see if it is able to run some codes
* Clone the repository then you can run in your own computer

### 5. Step by step instructions
- First: load the dataset files from the spambase/spambase.data
- Second: check the integrity of the dataset
- Third: split data for training and testing also rescale the data
- Fourth: train and test three SVM model (linear, sigmoid and polynomial)
- Fifth: train and test neural network model
- Sixth: print out result and show the accuracy of 4 models in the same bar chart

### 6. Results
* https://github.com/SuperSimonC/MSA-AI/blob/master/result.pdf
* From the chart in the website, it is obvious that the accuracy for SVM_linear and Neural Network are more than 90%, which performs better compare with other two.
