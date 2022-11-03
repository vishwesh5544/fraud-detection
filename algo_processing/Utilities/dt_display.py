def DT_Display(dt_gb_values, dt_Ab_values, dt_b_values):
    print("<----------------------------------------------------------------------------------------------------------->\n")
    print("Decision Tree \n")
    print("\t\t\t\t\t\t\t\t\t Gradient Boosted   \t\t\t\t\t  Ada Boosted \t\t\t\t\t Bagged \n")
    print("Precision Score       :              ", dt_gb_values[1][0], "\t\t\t\t\t", dt_Ab_values[1][0],"\t\t\t\t",dt_b_values[1][0], "\n")
    print("Recall Score          :              ", dt_gb_values[1][1], "\t\t\t\t\t", dt_Ab_values[1][1],"\t\t\t\t\t",dt_b_values[1][1],"\n")
    print("Accuracy Score        :              ", dt_gb_values[1][2], "\t\t\t\t\t", dt_Ab_values[1][2],"\t\t\t\t",dt_b_values[1][2],"\n")

