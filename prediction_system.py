import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import utils
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class PredictionModel:

    def __init__(self):
        df = self.load_data()
        self.model = self.train_model(df)
    
    def train_model(self, df):
        print("Training classifier.")
        X = df[list(set(df.columns) - set(["index", "decision"]))]
        y = df["decision"]
        self.columns = X.columns
        
        X_resampled, y_resampled = RandomOverSampler(ratio={"A":150, "B":100}).fit_sample(X, y)
        X_resampled, y_resampled = RandomUnderSampler(ratio={"None":250}).fit_sample(X_resampled, y_resampled)

        model = RandomForestClassifier(n_estimators=500, max_features=None)
        model.fit(X_resampled, y_resampled)

        print("Training complete.")
        return model

    def predict(self, row):
        def preprocess(row):
            row = row[self.columns]
            if len(row) == 1:
                return row.values.reshape(1, -1)
            return row
        row = preprocess(row)
        
        return self.model.predict(row)

    def load_data(self):
        df = utils.load_data("data/trialPromoResults.csv")
        return df