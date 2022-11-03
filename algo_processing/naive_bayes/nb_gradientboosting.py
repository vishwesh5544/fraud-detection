from algo_processing.Utilities.precision_recall_accu import PrecisionRecallAccuracy
from algo_processing.Utilities.confusion_matrix import ConfusionMatrix

def NB_GradientBoosting(X_train, X_test, Y_train, Y_test):

    from sklearn.ensemble import GradientBoostingClassifier

    gradboost = GradientBoostingClassifier(n_estimators=100, loss='exponential', learning_rate=1)

    gradboost.fit(X_train, Y_train)

    nb_pca = PrecisionRecallAccuracy(gradboost, X_train, X_test, Y_train, Y_test, "adaBoosting using (Naive Bayes)")
    nb_cm = ConfusionMatrix(gradboost, X_test, Y_test)

    mt_nb_Values = []
    mt_nb_Values.append(nb_cm)
    mt_nb_Values.append(nb_pca)

    return mt_nb_Values
