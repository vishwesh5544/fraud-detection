from models.Utilities.confusion_matrix import ConfusionMatrix
from models.Utilities.precision_recall_accu import PrecisionRecallAccuracy


def DecisionTree(X_train, X_test, Y_train, Y_test):

        from sklearn.tree import DecisionTreeClassifier
        DC_classifier = DecisionTreeClassifier(random_state=0, criterion='entropy')
        DC_classifier.fit(X_train, Y_train)

        DC_cm = (ConfusionMatrix(DC_classifier, X_test, Y_test))
        DC_pra = (PrecisionRecallAccuracy(DC_classifier, X_test, Y_test))

        DC_Values = []
        DC_Values.append(DC_cm)
        DC_Values.append(DC_pra)
        return DC_Values
