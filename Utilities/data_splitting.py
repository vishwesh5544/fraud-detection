from res.imports.Imports import *
from Utilities.feature_scaling import apply_feature_scaling


def DataSplitting(df2):
        # Splitting Data into Dependent and Independent variables.
        X = df2.iloc[:, :-1].values  # Independent
        pd.set_option('display.max_columns', None)
        Y = df2.iloc[:, -1].values  # Dependent

        # Splitting Data into Training set and Test set.
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
        print("Size of X_train: ", len(X_train), ", Size of X_test: ", len(X_test), X_train, X_test)
        print("Size of Y_train: ", len(Y_train), ", Size of Y_test: ", len(Y_test))

        print("<----------------------------------------------------------------------------------------------------------->")

        # Calling FeatureScaling Class
        li_Fs_Data = apply_feature_scaling(X_train, X_test, Y_train, Y_test)
        return li_Fs_Data

