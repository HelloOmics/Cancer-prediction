# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:21:37 2025

@author: roxan
"""

# change working directory
import os
os.chdir(r"C:\Users\roxan\OneDrive\Desktop\Masters_Biology_FU\DIY_data_science\datasets\Cancer_genes")

#----------------------------------------------------------------

# download the data from kaggle
import kagglehub
kagglehub.login()  # Authenticate if needed

# To browse Datasets using bash via python script
# need to use subprocess
import subprocess

# Run the kaggle command
result = subprocess.run(["kaggle", "datasets", "list", "-s", "Cancer Genomic"], capture_output=True, text=True)

# Print the output
print(result.stdout) # brsahan/genomic-data-for-cancer => dataset of interest

# to downoad dataset using bash via python script
# use subprocess
result2 = subprocess.run(["kaggle", "datasets", "download", "-d", "brsahan/genomic-data-for-cancer "],
                         capture_output=True, text=True)

# Print the output
print(result2.stdout)
# Dataset URL: https://www.kaggle.com/datasets/brsahan/genomic-data-for-cancer 
# License(s): apache-2.0
# downloaded file is now found in working directory
# Attention! must be unzipped!

# read in the data:
import pandas as pd
df = pd.read_csv(r"C:\Users\roxan\OneDrive\Desktop\Masters_Biology_FU\DIY_data_science\datasets\Cancer_genes\gene_expression.csv")


#----------------------------------------------------------------

# Explore the data:
    
df.head()
df.tail()
df.info() # integer = full number, float = comma
df.describe().T # 3.000 rows in total
df["Cancer Present"].value_counts() 
# Cancer Present
# 1    1500
# 0    1500 
# => dataset is well balanced, no bias

# Plot the data:
import plotly.express as px

df.columns
# Create Plot
fig = px.scatter(df, x='Gene One', y='Gene Two', title='Gene expression associated with Cancer', 
                 color='Cancer Present')

# Update Plot: Add titles, labels, or modify aesthetics.
fig.update_layout(
    xaxis_title="Gene One",
    yaxis_title="Gene Two",
    title_font=dict(size=20),
    template="plotly_dark" # applies dark theme
    )


# show your figure
fig.show() # spyder is having difficulty in rendering/showing the plot?

# open plot as an html file
fig.write_html("plot.html")

# Save the plot as a PNG image =>  2d image for Github README
fig.write_image("plot.png")

import webbrowser

# Path to the HTML file
file_path = "plot.html"

# Open the file in the default web browser
webbrowser.open(file_path)


#------------------------------------------------------------

# code from https://www.kaggle.com/code/muhammedaliyilmazz/comparing-ml-and-dl-models-for-accuracy-and-r2/notebook

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy

#------------------
# explore data
df.info() # basic info like column names and data types
df.describe() # basic statistics
df.isnull().sum # counts the naN

#---------------------

# define target column
target = "Cancer Present" 
X = df.drop(columns=[target]) # split labels from values
y = df[target] #separate dataset for labels

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=537) # or 42?

# standardize features:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#-----------------
# 1) Create a dictionary of Regression models
regressors = {
    "Linear Regression" : LinearRegression(),
    "Decision Tree" : DecisionTreeRegressor(),
    "Random Forest" : RandomForestRegressor(),
    "Gradient Boosting" : GradientBoostingRegressor(),
    "AdaBoost" : AdaBoostRegressor(),
    "SVR" : SVR(),
    "KNeighbors" : KNeighborsRegressor(),
 }

# 1 DecisionTreeRegressor() vs. DecisionTreeClassifier()
# DecisionTreeRegressor(): Used for regression problems, 
# where the target variable is continuous (e.g., predicting 
# house prices).
# DecisionTreeClassifier(): Used for classification problems, 
# where the target variable is categorical (e.g., classifying 
# emails as spam or not spam)

#In summary, regressors predict numeric values, while classifiers predict categories.


#RandomForestRegressor(): An ensemble of multiple decision trees that predict continuous values (regression).Accuracy
#GradientBoostingRegressor(): Uses boosting techniques to combine weak learners (decision trees) for continuous variable prediction.
#AdaBoostRegressor(): Boosting technique with weak regressors (usually decision trees) for regression tasks.
#KNeighborsRegressor(): Predicts a continuous value by averaging the values of its nearest neighbors.


# 2) make loop function that iterates of dictionary, fits each model
# to the training data, makes predictions, evaluates performance using R2 score
# and stores the results in a list
regression_results = []
for name, model in regressors.items(): # loop, name = "Lin Regre", model = Linear Regression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    regression_results.append((name, score))
    
# Calling .items() on a dictionary returns an iterable of tuples where:
# The first element (name) is the key (e.g., 'Linear Regression').
# The second element (model) is the value (e.g., LinearRegression()).
# Each iteration assigns:
# name → 'Linear Regression', 'Decision Tree', etc.
# model → LinearRegression(), DecisionTreeRegressor(), etc
# .items() provides (name, model) pairs for iteration

# Tuple vs List:
# 1. Mutability
# Tuple (tuple): Immutable – Once created, the elements cannot be changed, added, or removed.
# List (list): Mutable – You can modify, add, or remove elements after creation.
# my_tuple = (1, 2, 3)
# my_list = [1, 2, 3]

# 2. Performance
# Tuples faster than lists for iteration and lookup 
# because they are immutable --> more optimized for performance.
# If you're working with a large dataset that doesn’t require modifications, tuples can provide a slight speed boost.

# 3. Usage & Purpose
# Tuple: Used when data should remain unchanged (e.g., coordinates (x, y), database records, fixed settings).
# List: Used when data will change (e.g., shopping lists, dynamic arrays, collections that need sorting).

#----------------

# Display Regression results
regression_df = pd.DataFrame(regression_results, columns=["Model", "R2 Score"])
regression_df


# visualize model R2 scores
sns.barplot(x="Model", y="R2 Score", data=regression_df, palette="rainbow")
plt.xticks(rotation=45)
plt.show()

#--------- Classification models

le = LabelEncoder()
# LabelEncoder() is a tool from sklearn.preprocessing that converts 
# categorical labels into numerical values
y = le.fit_transform(y)
# Fits the encoder to the labels in y
# mapping each unique category to a number.
# Transforms the original y values into these encoded numerical labels.
print(y.dtype) 
# our values are already an integer, so technically don't need to 
# turn into integers via LabelEncoder()
# However!!
# good for consistency in ML pipelnes
# good in case you may add non binary data later expand data set
# goes from binary to three options = ensures consitent encoding
# good to avoid possible data type issues:
# Even though your labels are already numerical
# they might be stored as categorical (object or bool) in pandas. 
# LabelEncoder() converts them explicitly to an integer format, 
# avoiding potential issues when working with libraries
# that expect numerical labels. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train_test_split() is used to split the dataset into a 
# training set and a testing set.
# test_size=0.2: 20% of the data will be allocated to testing, 
# while 80% will be used for training.
# random_state=42: Ensures reproducibility by using the same random seed every time.

# After the split:
# X_train: Features for training.
# X_test: Features for testing.
# y_train: Labels corresponding to X_train.
# y_test: Labels corresponding to X_test.



classifiers = {
    "Logistic Regression" : LogisticRegression(),
    "Decision Tree" : DecisionTreeClassifier(),
    "Random Forest" : RandomForestClassifier(),
    "Gradient Boosting" : GradientBoostingClassifier(),
    "AdaBoost" : AdaBoostClassifier(),
    "SVC" : SVC(),
    "KNeighbors" : KNeighborsClassifier(),
    "Naive Bayes" : GaussianNB()
    }

classification_results = []
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    classification_results.append((name, score))
    
# display classififcation results
classification_df = pd.DataFrame(classification_results, columns=["Model", "Accuracy"])
classification_df

# visualize the models accuracy
sns.barplot(x="Model", y="Accuracy", data=classification_df, palette="rainbow")
plt.xticks(rotation=45) 
plt.show()

#----------- Build Deep learning models --------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


input_shape = X.shape[1] # input_shape now contains number of columns in dataset X
# X and y were established/split before = dataset
# X.shape returns the dimensions of X as a tuple (num_samples, num_features)
# X.shape[0] gives the number of rows (data samples)
# X.shape[1] gives the number of columns (features)

dl_models = [] # create list for models
dl_results = [] # create list for results

for units in [(16,), (32, 16), (64, 32, 16)]: # units is a variable name our models
# could also call it layer_sizes or architecture
# values inside the list define the number of neurons in each hidden layer
    model = Sequential()
    model.add(Input(shape=(input_shape,))) # model.add(...) → We are adding layers to our Sequential model.
# defines the input layer with a shape equal to X.shape[1], meaning the number of features (columns) in dataset X.
# For example, if X has 20 columns, then input_shape = 20
# This ensures the network knows how many inputs to expect from the dataset

#A Sequential model in Keras = simple type of neural network 
# where layers are stacked in a linear order, meaning data flows from one layer 
# to the next sequentially.
# like stacking LEGO blocks one on top of another, where each layer’s output feeds into the next layer.

# example:
# model = Sequential([
#   Dense(16, activation="relu"),
#   Dense(8, activation="relu"),
#   Dense(1, activation="sigmoid")
# ])
# explanation:
# First hidden layer → 16 neurons, ReLU activation.
# Second hidden layer → 8 neurons, ReLU activation.
# Output layer → 1 neuron, Sigmoid activation (for binary classification).

# Sequential models are great for simple feedforward networks, but some tasks require more flexibility.
# Here are some alternatives:
# Functional API → When layers have multiple inputs/outputs or share connections.
# Subclassing Model API → When defining completely custom architectures.
# Convolutional Neural Networks (CNNs) → Best for image recognition.
# Recurrent Neural Networks (RNNs) → Best for sequential data like time series or language modeling.
# Transformer Models → Used in advanced language models (like ChatGPT or Copilot 😉).
# If your task involves complex connections or multiple pathways, the Functional API is often the better choice.

for u in units: 
# units was previously defined and is a variable that contains our model architecture
# u is a loop variable that takes values from units one at a time. So when looping:
# First iteration: u = (16,)
# Second iteration: u = (32, 16)
# Third iteration: u = (64, 32, 16)
        model.add(Dense(u, activation="relu")) # u = number of neurons, value taken from units loop
        model.add(Dense(1, activation="sigmoid"))
        
# activation function transforms the output of each neuron to introduce non-linearity
# making the network capable of learning complex patterns
# Each neuron receives an input, performs some computations + applies an activation function
# before passing the value to the next layer.

# ReLU (relu) → Best for hidden layers = Stands for Rectified Linear Unit.
# It replaces negative values with zero but keeps positive values the same.
# Helps prevent the vanishing gradient problem in deep networks.
# Commonly used for hidden layers because it speeds up training and improves stability.

# Sigmoid (sigmoid) → Best for output layer (binary classification)
# It transforms values into the range (0,1), making it perfect for classification tasks.
# If the output is close to 1, the model predicts class A.
# If the output is close to 0, the model predicts class B.
# Used for binary classification problems 
        
        model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        dl_models.append(model)
        dl_results.append((f"DL Model {len(units)} Layers", accuracy))
        print(f"DL Model with {len(units)} Layers : Accuracy = {accuracy:.4f}")
        
# display deep learning results:
    dl_results_df = pd.DataFrame(dl_results, columns=["Model", "Accuracy"])
    dl_results_df

# visualise the accuracy of the 3 deep learning models:
    sns.barplot(x="Model", y="Accuracy", data=dl_results_df, palette="rainbow")
    plt.xticks(rotation=45)
    plt.show()
    
# Model with 3 Layers is the most effective


#-------------------------------------------

# explanation to loops structure:
    
    for item in collection:
    # Do something with item
    
# item → Represents each individual element in the collection.
# collection → Can be a list, tuple, dictionary, range, or any iterable object.
# The loop runs once for each element in collection, assigning it to item each time.

# example:
    fruits = ["apple", "banana", "cherry"]

for fruit in fruits:
    print(f"Fruit: {fruit}") 
# f string allows you to add variables straight into written strings
# now it will treat {fruits} as a variable (to be filled) instead of a literral text


    