# Credit Card Fraud Detection using Machine Learning in Python | Machine Learning Projects

'''Word Flow Steps :- 
   Credit Card Data --> Data Prepocessing --> Data analysis --> Train Test Split -->  Logistic Regression --> Evaluation'''
   
      
#Importing the Dependencies
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('/content/creditcard.csv')

#first 5 rows of dataset
credit_card_data.head()

#last 5 rows of dataset
credit_card_data.tail()

#dataset informations
credit_card_data.info()

#Checking the number of  missing values in each column
credit_card_data.isnull().sum()

#distribution of legit transactions and fraudulent transactions
credit_card_data['Class'].value_counts() 

'''This Dataset is highly unbalanced
    
    0 --> Normal Transaction
    1 --> fraudulent Transaction'''
    
#Separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

#statistical measures of the data
legit.Amount.describe()

fraud.Amount.describe()

# compare the values for both transactions
credit_card_data.groupby('Class').mean()

# Under-Sampling

'''Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transaction
   Number of Fraudlent Transactions --> 492'''
   
legit_sample = legit.sample(n = 492)

# Concatenating two DataFrames

new_dataset = pd.concat([legit_sample , fraud], axis = 0)    
# conclusion :- axis = 0 means "rows" and axis = 1 means "columns"

new_dataset.head()

new_dataset.tail()

new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

#Spliting the data into Features & Targets
X = new_dataset.drop(columns = 'Class' , axis = 1)
Y = new_dataset['Class']

print(X)

print(Y)

#Split the data into Training data & Testing Data
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size = 0.2 , stratify = Y , random_state = 2)

print(X.shape, X_train.shape, X_test.shape)

# Model training

# Logistic Regression
model = LogisticRegression()

# Training the Logistics Regression Model with Training Data
model.fit(X_train, Y_train)

# Model Evaluation

# Accuracy Score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy of Training data:- " , training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction , Y_test)

print("Accuracy score on Test Data :- " , test_data_accuracy)

# Note :- If there is a differnce in the accuracy of training data and test data the our model either be "underfitted" or "overfitted".