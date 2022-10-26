import numpy as np
import pandas as pd

### <--- Data PreProcessing Phase. --->
# Reading data from dataset.
df = pd.read_csv('res/dataset/Book1.csv')

# Displaying Columns name to Work on, Alongside with Dataset Dimensionality.
print("column Names of Dataset :")
print(df.columns)
col_names = list(df)
print("<----------------------------------------------------------------------------------------------------------->")

# Converting data into Dataframe
# df = pd.DataFrame(df)
print("rows X columns : ", len(df), "X", len(col_names))

print("<----------------------------------------------------------------------------------------------------------->")

# Checking for Null values in the dataset.
print("Null count Table for the dataset ", df.isnull().sum().sum(), "\n")
print(df.isnull().sum())

print("<----------------------------------------------------------------------------------------------------------->")

# Dropping Rows with Null values.
df1 = df.dropna(axis=0, how="any", thresh=None, subset=None)
print("Rows dropped : ", df.isnull().sum().sum())
print(df1)

print("<----------------------------------------------------------------------------------------------------------->")

# Checking for Any Duplicate values in the dataset.
print("Count of Duplicate Values : ", df1.duplicated().sum())

print("<----------------------------------------------------------------------------------------------------------->")

# Removing Duplicate rows from Dataframe.
df2 = df1.drop_duplicates(keep=False)
print("Removing Duplicate Rows ... ")

# After removing duplicate rows.
print("Count of Duplicate Values : ", df2.duplicated().sum())
print("Count of Null values : ", df2.isnull().sum().sum(), "\n")
print("Count of Null values : \n", df2.isnull().sum(), "\n")
print("Number of unique Rows : ", len(df2), "\n")
print("Data Cleaning Completed...")

print("<----------------------------------------------------------------------------------------------------------->")

# Splitting Data into Dependent and Independent variables.
X = df2.iloc[:, :-1].values                    # Independent
pd.set_option('display.max_columns', None)
Y = df2.iloc[:, -1].values                     # Dependent
# print(X, "\n", Y)


# Splitting Data into Training set and Test set.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
print("Size of X_train: ", len(X_train),", Size of X_test: ", len(X_test), X_train ,X_test)
print("Size of Y_train: ", len(Y_train),", Size of Y_test: ", len(Y_test))

# Applying Feature Scaling
from sklearn.preprocessing import StandardScaler      # range ( ) , normalisation (-1,1)
sc = StandardScaler()
scale_col =['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']
X_train[:, 0:3] = sc.fit_transform(X_train[:, 0:3])
X_test[:, 0:3] = sc.transform(X_test[:, 0:3])
print("Training Independent Data : \n ", X_train, "\n", "Testing Independent Data \n ", X_test)
print("Training Independent Data : \n ", Y_train, "\n", "Testing Independent Data \n ", Y_test)

print("<----------------------------------------------------------------------------------------------------------->")





### <--- Model Selection Phase. --->


### <--- Optimizing Models Phase. --->