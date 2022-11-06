
def PrecisionRecallAccuracy(classifier, X_train, X_test, Y_train, Y_test, model, CV=10):

# Computing Accuracy with K-fold Cross Validation.

        from sklearn.model_selection import cross_val_score
        accuracy = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=CV)

# Predicting Test Set Value
        y_pred = classifier.predict(X_test)

# Calculating Precision Score.
        from sklearn.metrics import precision_score
        ps = precision_score(Y_test, y_pred, pos_label=1)           # pos_label [2,4]


# Calculating Recall Score.
        from sklearn.metrics import recall_score, accuracy_score
        rs = recall_score(Y_test, y_pred, pos_label=1)


# Calculating Accuracy Score.
        ac = accuracy_score(Y_test, y_pred)

# Calculating F1_score
        from sklearn.metrics import f1_score
        f1 = f1_score(Y_test, y_pred)


        mt_pra = [ps, rs, ac, f1]
        return mt_pra
