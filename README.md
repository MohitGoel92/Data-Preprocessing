# Data Preprocessing

In any Machine Learning process, data preprocessing is a crucial step in which the data gets cleaned, encoded and at times reduced to bring it to a state where the machine can easily process it. In other words, the features of the data can now be easily interpreted by the algorithm, therefore preparing for the long journey ahead. If we do not perform the data preprocessing step, the machine learning model may not work properly. There are four main steps to data preprocessing, they include: taking care of missing data, encoding categorical data, splitting the dataset into the training set and test set, and feature scaling.

## Missing Data

When we have missing numerical data, we have two main approaches to this situation. The first approach is to remove the rows with the missing data of the large dataset as removing, for instance, 1% of the dataset should not harm the learning quality of our model. In the second approach, by default we fill the gaps by calculating the average value for the column. This goes for all the features with missing data in the dataset.

## Encoding Categorical Data

Categorical data is data that contains columns with categories, for instance Country Names, Colours, Yes/No ... etc. We encode the text (categories) into numbers so we can use them in equations. We use the "OneHotEncoder" library to create binary vectors for each category, for example: Blue is (1,0,0), Red is (0,1,0) and Green is (0,0,1). We do not assign each category a number, for instance: Blue is 1, Red is 2, and Green is 3, as these numbers are ordinal. Therefore, this may imply that Green > Red > Blue as 3 > 2 > 1, however, this makes no sense and therefore it is better to assign each category a binary vector. 

The code below is an example of encoding categorical data:

```
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X = LabelEncoder()
X[:,0] = le_X.fit_transform(X[:,0])
ohe_X = OneHotEncoder(categorical_features = [0])
X = ohe_X.fit_transform(X).toarray()
```

## Splitting the dataset into the training set and test set

The training set is the dataset we build our model on and the test set is the dataset we are testing the performance of our machine learning model on. In theory, the performance of the test set should not be that different in comparison to the performance of the training set. This suggests a good correlation being detected by the machine learning model and a good understanding from the machine. As a rule of thumb, the size of the test set is usually between 0.2 and 0.4 (20%-40%).

## Feature Scaling

Machine Learning models are based on Euclidean Distances. The below diagram illustrates Euclidean Distances:

<img src = 'Euclidean Distances Diagram.png' width='350'>

If the features (variables/columns) have a very different range of values in comparison to other features, the feature with the larger range will outweigh the column with the smaller range. In this case, it will be as though the feature with the smaller range does not exist in the machine learning equations. We therefore transform the features onto the same scale, resulting in all values being within a short range. If the dependent variable also takes a huge range of values, we will apply feature scaling to the dependent variable as well.

**Note:** If the Machine Learning models, for instance, Artificial Neural Networks or Decision Trees (ensemble learning) are not based on Euclidean Distances, we still need to apply feature scaling as the algorithm will converge must faster. If we do not apply Feature Scaling, our models may run for a very long time.

When applying feature scaling, we may either use Standardisation or Normalisation, both are given below:

<img src = 'Formulas.png' width='350'>

When using Standardisation, the resulting values in the column will approximately be between -3 and 3. When using Normalisation, the resulting values will be between 0 and 1.

**Note:** Normalisation will be used only when our features mostly follow a normal distribution. Standardisation however can be used all the time and it will work.

Feature scaling is applied after splitting the dataset into the training set and test set seperately. The test set is supposed to be a brand new set on which we evaluate our machine learning model on. We therefore do not work with the test set while the machine learning model is training. As we are taking the "Mean(X)" and "Standard Deviation(X)", if we apply feature scaling before the split we will be taking the Mean(X) and Standard Deviation(X) of the whole dataset (the training set and test set inclusive), of which we are not suppose to have as the test dataset is representing new or future data. Therefore, the main reason why feature scaling is applied after the splitting of the dataset into the training and test set is to prevent information leakage on the test set which we are not meant to have until the training of the machine learning model has been completed.

The code below is an example of splitting the dataset into the training set and test set, followed by feature scaling:

```
# Splitting the dataset into the training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```

In the above code we observe that the StandardScaler is fitted to X_train but it is only transformed onto X_test. This is because "fit" gets the value for Mean(X) and Standard Deviation(X) of each feature, and these metrics that are taken from the training dataset are applied in transforming the values in the test dataset. If "fit_transform" is applied to the test set as well, the feature scaling will use the Mean(X) and Standard Deviation(X) from the test set and not the training set.

**Note:** We do not apply feature scaling to our encoded categorical variables as they are already within the same range. Also, due to the binary vectors corresponding to a categorical name, we will lose interpretability. Therefore, the feature scaler not helping with performance either. 

## References

Euclidean Distances diagram: https://en.wikipedia.org/wiki/Euclidean_distance

Online LaTex writer: https://www.codecogs.com/latex/eqneditor.php
