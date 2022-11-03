from models.Utilities.confusion_matrix import ConfusionMatrix
from models.Utilities.precision_recall_accu import PrecisionRecallAccuracy

def NaiveBayes(X_train, X_test, Y_train, Y_test):

        from sklearn.naive_bayes import GaussianNB

        NB_classifier = GaussianNB()
        NB_classifier.fit(X_train, Y_train)

        NB_cm = (ConfusionMatrix(NB_classifier, X_test, Y_test))
        NB_pra = (PrecisionRecallAccuracy(NB_classifier, X_test, Y_test))

        NB_Values =[]
        NB_Values.append(NB_cm)
        NB_Values.append(NB_pra)

        return NB_Values