# Data Preprocessing

# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset

ds = pd.read_csv('Data.csv')
X = ds.iloc[:,:-1].values
y = ds.iloc[:,-1].values

# Taking care of missing data

# When we have missing numerical data, by default we fill the gaps by calculating the average values of the column.
# This goes for all features in the dataset.

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

X[:,1:3] = imputer.fit_transform(X[:,1:3])

# Encoding categorical data

# Categorical data is the data which contains columns that contains categories (e.g. Countries, Names, Yes/No ...).
# We encode the text (categories) into numbers so we can use them in our equations. Each category is given a number
# but as each category will be assigned a different number, this may imply one variable is better/higher ranked than 
# another (e.g. If green = 1 and blue = 2, does this mean blue > green?). We therefore make use of dummy variables
# and use the OneHotEncoder.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X = LabelEncoder()
le_y = LabelEncoder()

X[:,0] = le_X.fit_transform(X[:,0])

y = le_y.fit_transform(y)

ohe_X = OneHotEncoder(categorical_features = [0])
X = ohe_X.fit_transform(X).toarray()

# Avoiding the dummy variable trap

X = X[:,1:]

# Splitting the dataset into the training set and test set

# The training set is the set we build our model on and the testing set is the set we test the performance of our
# machine learning model on. In theory, the performance of the test set shouldn't be that different in comparison
# to the performance of the training set. This suggests a good correlation detected by the machine learning model
# and a good understanding from the machine.

# We have used a test size of 0.3 (3/10 or 30%)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the Logistic Reggresion to the dataset

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predicting the test set results

y_pred = lr.predict(X_test)

# Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
