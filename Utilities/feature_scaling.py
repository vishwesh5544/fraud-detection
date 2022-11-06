from res.imports.Imports import *

def apply_feature_scaling(X_train, X_test, Y_train, Y_test):

        # Applying Feature Scaling.
        sc = StandardScaler()
        x_train = x_train[:, 0:3] = sc.fit_transform(X_train[:, 0:3])
        x_test = x_test[:, 0:3] = sc.transform(X_test[:, 0:3])

        li_Fs_Data = (X_train, X_test, Y_train, Y_test)
        return li_Fs_Data



