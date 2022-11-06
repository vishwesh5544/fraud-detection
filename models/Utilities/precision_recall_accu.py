def PrecisionRecallAccuracy(classifier, X_test, Y_test):
    # Predicting Test Set Value.
    y_pred = classifier.predict(X_test)

    # Calculating Precision Score.
    from sklearn.metrics import precision_score
    ps = precision_score(Y_test, y_pred, pos_label=0)

    # Calculating Recall Score.
    from sklearn.metrics import recall_score, accuracy_score
    rs = recall_score(Y_test, y_pred, pos_label=0)

    # Calculating Accuracy score.
    ac = accuracy_score(Y_test, y_pred)

    # Calculating F1_score
    from sklearn.metrics import f1_score
    f1 = f1_score(Y_test, y_pred)

    par_Values = [ps, rs, ac, f1]
    return par_Values
