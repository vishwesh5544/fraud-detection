def NB_Display(nb_gb_values, nb_Ab_values, nb_b_values):
    print("<----------------------------------------------------------------------------------------------------------->\n")
    print("Naive Bayes \n")
    print("\t\t\t\t\t\t\t\t\t Gradient Boosted   \t\t\t\t\t  Ada Boosted \t\t\t\t\t Bagged \n")
    print("Precision Score       :              ", nb_gb_values[1][0], "\t\t\t\t", nb_Ab_values[1][0],"\t\t\t\t\t\t\t\t",nb_b_values[1][0], "\n")
    print("Recall Score          :              ", nb_gb_values[1][1], "\t\t\t\t", nb_Ab_values[1][1],"\t\t\t\t",nb_b_values[1][1], "\n")
    print("Accuracy Score        :              ", nb_gb_values[1][2], "\t\t\t\t", nb_Ab_values[1][2],"\t\t\t\t",nb_b_values[1][2], "\n")
