import pandas as pd
import numpy as np

class CsvPredictionModel:

    def __init__(self, filepath):
        self.model = pd.read_csv(filepath, index_col="index")

    def predict(self, df):
        pred = []
        for i in range(len(df)):
            index = df.loc[i]["index"]
            pred.append(self.model.loc[index]["status"])
        return pred
