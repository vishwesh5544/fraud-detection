from res.imports.Imports import *
from Utilities.remove_null import Remove_Null


def Fetching_Data():
    # Reading data from dataset.

    df = pd.read_csv('res/dataset/Book1.csv')

    # Displaying Columns name to Work on, Alongside with Dataset Dimensionality.
    print("column Names of Dataset :")
    print(df.columns)
    col_names = list(df)
    print(
        "<----------------------------------------------------------------------------------------------------------->")

    # Converting data into Dataframe.
    print("rows X columns : ", len(df), "X", len(col_names))

    print(
        "<----------------------------------------------------------------------------------------------------------->")

    # Calling Remove_Null function.
    li_Fs_Data = Remove_Null(df)
    return li_Fs_Data
