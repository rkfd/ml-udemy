# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# import dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# take care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3]) # columns with numerical values
x[:,1:3] = imputer.transform(x[:,1:3])

# encoding independent variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# encoding the dependent variable
le = LabelEncoder()
y = le.fit_transform(y)

# splitting the data into train and test set
x_train, x_test, y_train, y_test = train_test_split(x,  y, test_size=0.2, random_state=1)

# feature scaling
sc = StandardScaler() # standardisation
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:] = sc.fit_transform(x_test[:,3:])