from algo_processing.Utilities.precision_recall_accu import PrecisionRecallAccuracy
from algo_processing.Utilities.confusion_matrix import ConfusionMatrix


def NbAdaBoosting(X_train, X_test, Y_train, Y_test):

        from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
        classifier = RandomForestClassifier(random_state=0, criterion='entropy', n_estimators=100)

        adaboost = AdaBoostClassifier(n_estimators=100, base_estimator=classifier, learning_rate=1)
        adaboost.fit(X_train, Y_train)

        dt_pca = PrecisionRecallAccuracy(adaboost, X_train, X_test, Y_train, Y_test, "adaBoosting using (Naive Bayes)")
        dt_cm = ConfusionMatrix(adaboost, X_test, Y_test)

        mt_dt_Values = []
        mt_dt_Values.append(dt_cm)
        mt_dt_Values.append(dt_pca)

        return mt_dt_Values