from res.imports.Imports import *

def PrecisionRecallAccuracy(classifier, X_train, X_test, Y_train, Y_test, model, CV=10):

# Computing Accuracy with K-fold Cross Validation.

        from sklearn.model_selection import cross_val_score
        accuracy = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=CV)

        This = model
        # print(f"Accuracy For {This} Model : " + "{: .2f} %".format(accuracy.mean() * 100))

# Predicting Test Set Value.
        y_pred = classifier.predict(X_test)

# Calculating Precision Score.
        from sklearn.metrics import precision_score
        ps = precision_score(Y_test, y_pred, pos_label=1)           # pos_label [2,4]
        # print(f"Precision Score : {ps} ")

# Calculating Recall Score.
        from sklearn.metrics import recall_score, accuracy_score
        rs = recall_score(Y_test, y_pred, pos_label=1)
        # print(f"Recall score : {rs} " + "\n")

# Calculating Accuracy Score.
        ac = accuracy_score(Y_test, y_pred)

        mt_pra = [ps, rs, ac]
        return mt_pra
