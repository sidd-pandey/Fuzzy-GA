import pandas as pd
from utils import load_data, expected_profit_customer
from tqdm import tqdm
import numpy as np
from prediction_system import PredictionModel
from expert_system import ExpertSystem


# Read the data frame
df = load_data("data/custdatabase.csv")

model = PredictionModel()
product_predict = model.predict(df)

expert = ExpertSystem(df)

# Compute expected profit for each customer
def expected_profit_all_customers(df):    
    cust_predict = []
    for index in tqdm(range(len(df))): 
        row = df.loc[index]
        cip = expert.predict(row)
        product = product_predict[index]
        expected_profit = expected_profit_customer(cip, product)
        cust_predict.append([df.loc[index, "index"], product, cip, expected_profit])   
    
    return cust_predict
 
cust_predict = expected_profit_all_customers(df)
cust_predict_df = pd.DataFrame(cust_predict, columns=["index", "product", "cip", "expected profit"])
cust_predict_df.to_csv("data/Cust_Predict.csv", index=False)

cust_predict_df_sorted = cust_predict_df.sort_values(by=["expected profit"], ascending=False)
cust_campaign_400 = cust_predict_df_sorted[:400]
cust_campaign_400.to_csv("data/Cust_Predict_400.csv", index=False)

expected_profit_campaign = np.sum(cust_campaign_400["expected profit"].values)
print("Predicted:", expected_profit_campaign)

# cust_actual = pd.read_csv("data/Cust_Actual")

