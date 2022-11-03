
def PrecisionRecallAccuracy(classifier, X_test, Y_test):

# Predicting Test Set Value.
        y_pred = classifier.predict(X_test)

# Calculating Precision Score.
        from sklearn.metrics import precision_score
        ps = precision_score(Y_test, y_pred, pos_label=0)           # pos_label [2,4]
        # print(f"Precision Score       :                {ps} ")

# Calculating Recall Score.
        from sklearn.metrics import recall_score, accuracy_score
        rs = recall_score(Y_test, y_pred, pos_label=0)
        # print(f"Recall score          :                {rs} ")

        ac = accuracy_score(Y_test, y_pred)

        # print(f"Accuracy Value        :                {ac} ")
        par_Values = [ps, rs, ac]
        return par_Values

