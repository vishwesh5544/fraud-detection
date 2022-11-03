from algo_processing.Utilities.precision_recall_accu import PrecisionRecallAccuracy
from algo_processing.Utilities.confusion_matrix import ConfusionMatrix

def DT_GradientBoosting(X_train, X_test, Y_train, Y_test):

    from sklearn.ensemble import GradientBoostingClassifier

    gradboost = GradientBoostingClassifier(n_estimators=100, loss='exponential', criterion='squared_error', learning_rate=1)

    gradboost.fit(X_train, Y_train)

    dt_pca = PrecisionRecallAccuracy(gradboost, X_train, X_test, Y_train, Y_test, "adaBoosting using (Naive Bayes)")
    dt_cm = ConfusionMatrix(gradboost, X_test, Y_test)

    mt_dt_Values = []
    mt_dt_Values.append(dt_cm)
    mt_dt_Values.append(dt_pca)

    return mt_dt_Values