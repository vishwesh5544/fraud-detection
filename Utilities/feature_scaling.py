from res.imports.Imports import *
from models.display import Display

def apply_feature_scaling(X_train, X_test, Y_train, Y_test):

        # Applying Feature Scaling.
        sc = StandardScaler()
        x_train = x_train[:, 0:3] = sc.fit_transform(X_train[:, 0:3])
        x_test = x_test[:, 0:3] = sc.transform(X_test[:, 0:3])
        # print("Training Independent Data : \n ", self.x_train, "\n", "Testing Independent Data \n ", self.x_test)
        # print("Training Independent Data : \n ", self.Y_train, "\n", "Testing Independent Data \n ", self.Y_test)

        # Display(self.x_train, self.x_test, self.Y_train, self.Y_test)
        # NaiveBayes(self.x_train, self.x_test, self.Y_train, self.Y_test)

    # def GetVal(self):
        li_Fs_Data = (X_train, X_test, Y_train, Y_test)
        return li_Fs_Data



