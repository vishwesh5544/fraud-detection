from models.Utilities.modules_imp import *

class Code:
    def __init__(self):
        super().__init__()

        ### <--- Data PreProcessing Phase. --->
        """
        1. Starting with Fetching data.
        2. Dealing with Removing Null Values.
        3. Dealing with Duplicate Values.
        4. Splitting Data.
        5. Applying Feature Scaling.  
        """

        dataF = Fetching_Data()
        print("Featured scaled Training Independent : \n", dataF[0], "\n")
        print("Featured scaled Test Independent : \n", dataF[1], "\n")
        print("Featured scaled Training Dependent : \n", dataF[2], "\n")
        print("Featured scaled Test Dependent : \n", dataF[3], "\n")

        ### <--- Model Selection Phase. --->

        NB_Values = list(NaiveBayes(dataF[0], dataF[1], dataF[2], dataF[3]))
        DT_Values = list(DecisionTree(dataF[0], dataF[1], dataF[2], dataF[3]))

        ### <--- Display Section. --->

        Display(NB_Values, DT_Values)

        ### <--- Model Optimization --->
        print("Applying Optimization Techniques. \n")

        ## 1. Applying Gradient Boosting on Naive Bayes.

        nb_gb_values = NB_GradientBoosting(dataF[0], dataF[1], dataF[2], dataF[3])
        dt_gb_values = DT_GradientBoosting(dataF[0], dataF[1], dataF[2], dataF[3])

        ## 2. Applying Ada Boosting on Naive Bayes.

        nb_Ab_values = NbAdaBoosting(dataF[0], dataF[1], dataF[2], dataF[3])
        dt_Ab_values = DtAdaBoosting(dataF[0], dataF[1], dataF[2], dataF[3]
                                     )
        ## 3. Applying Bagging on Naive Bayes.

        nb_b_values = NbBagging(dataF[0], dataF[1], dataF[2], dataF[3])
        dt_b_values = DtBagging(dataF[0], dataF[1], dataF[2], dataF[3])

        ### <--- Displaying Optimization Results. --->

        NB_Display(nb_gb_values, nb_Ab_values, nb_b_values)
        DT_Display(dt_gb_values, dt_Ab_values, dt_b_values)


