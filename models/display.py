def Display(NB_Values, DT_Values):

        print("<----------------------------------------------------------------------------------------------------------->\n")
        print("\t\t\t\t\t\t\t\t\t   Naive Bayes   \t\t\t\t\t  Decision Tree \n")
        print("True Positive Values  :                  ", NB_Values[0][0], "\t\t\t\t\t\t\t\t", DT_Values[0][0], "\n")
        print("True Negative Values  :                  ", NB_Values[0][1], "\t\t\t\t\t\t\t\t", DT_Values[0][1], "\n")
        print("False Positive Values :                  ", NB_Values[0][2], "\t\t\t\t\t\t\t\t", DT_Values[0][2], "\n")
        print("False Negative Values :                  ", NB_Values[0][3], "\t\t\t\t\t\t\t\t", DT_Values[0][3], "\n")
        print("Precision Score       :                  ", NB_Values[1][0], "\t\t\t", DT_Values[1][0], "\n")
        print("Recall Score          :                  ", NB_Values[1][1], "\t\t\t", DT_Values[1][1], "\n")
        print("Accuracy Score        :                  ", NB_Values[1][2], "\t\t\t", DT_Values[1][2], "\n")
        print("F1_Score              :                  ", NB_Values[1][3], "\t\t\t", DT_Values[1][3], "\n")

        print("<----------------------------------------------------------------------------------------------------------->\n")




