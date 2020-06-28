import pandas as pd
from IPython.display import display

def showStatistic(data, features):
    print("+++++++++++ Show all Statistic +++++++++++++++")
    df_f = pd.DataFrame(data, columns=["importance"])
    df_f["labels"] = features
    df_f.sort_values("importance", inplace=True, ascending=False)
    display(df_f.head(10))
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    return
