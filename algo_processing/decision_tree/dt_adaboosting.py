from algo_processing.Utilities.precision_recall_accu import PrecisionRecallAccuracy
from algo_processing.Utilities.confusion_matrix import ConfusionMatrix


def DtAdaBoosting(X_train, X_test, Y_train, Y_test):

        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        classifierr = DecisionTreeClassifier()

        adaboost = AdaBoostClassifier(n_estimators=100, base_estimator=classifierr, learning_rate=1)
        adaboost.fit(X_train, Y_train)

        dt_pca = PrecisionRecallAccuracy(adaboost, X_train, X_test, Y_train, Y_test, "adaBoosting using (Naive Bayes)")
        dt_cm = ConfusionMatrix(adaboost, X_test, Y_test)

        mt_dt_Values = []
        mt_dt_Values.append(dt_cm)
        mt_dt_Values.append(dt_pca)

        return mt_dt_Values