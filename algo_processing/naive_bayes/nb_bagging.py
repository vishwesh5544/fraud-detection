from algo_processing.Utilities.precision_recall_accu import PrecisionRecallAccuracy
from algo_processing.Utilities.confusion_matrix import ConfusionMatrix


def NbBagging(X_train, X_test, Y_train, Y_test):

        from sklearn.model_selection import KFold, cross_val_score

        seed = 7
        kfold = KFold(n_splits=7, random_state=seed, shuffle=True)

        from sklearn.tree import DecisionTreeClassifier
        dc = DecisionTreeClassifier()
        num_tress = 20

        from sklearn.ensemble import BaggingClassifier
        Bagg_classifier = BaggingClassifier(base_estimator=dc, n_estimators=num_tress, random_state=seed)
        Bagg_classifier.fit(X_train, Y_train)

        mt_nb_pca = PrecisionRecallAccuracy(Bagg_classifier, X_train, X_test, Y_train, Y_test, "Bagging using (Decision Tree)", kfold)
        mt_nb_cm= ConfusionMatrix(Bagg_classifier, X_test, Y_test)

        mt_nb_values = []
        mt_nb_values.append(mt_nb_cm)
        mt_nb_values.append(mt_nb_pca)

        return mt_nb_values
