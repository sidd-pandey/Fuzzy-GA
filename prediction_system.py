import numpy as np
import pandas as pd
import sklearn.metrics as metrics
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
            return row.values.reshape(1, -1)
        row = preprocess(row)
        
        return self.model.predict(row)

    def load_data(self):
        df = pd.read_csv("data/trialPromoResults.csv")
        sex_map = {"M": 0, "F": 1}
        mstatus_map = {"single":0, "married":1, "widowed":2, "divorced":3}
        occupation_map = {'legal':0, 'IT':1, 'government':2, 'manuf':3, 'retired':4, 
                        'finance':5,'construct':6, 'education':7, 'medicine':8}
        education_map = {'postgrad':3, 'secondary':0, 'tertiary':1, 'professional':2}
        df["sex"] = df["sex"].map(sex_map)
        df["mstatus"] = df["mstatus"].map(mstatus_map)
        df["occupation"] = df["occupation"].map(occupation_map)
        df["education"] = df["education"].map(education_map)
        return df