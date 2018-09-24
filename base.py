from expert_system import expert_system
from prediction_system import PredictionModel
import pandas as pd
import utils
from tqdm import tqdm
import numpy as np

# Read the data frame
df = utils.load_data("data/custdatabase.csv")

model = PredictionModel()

# Compute expected profit for each customer
def expected_profit_all_customers(df):
    
    simulator = expert_system(df)

    # Expected profit for given customer
    def expected_profit_customer(cip, product):
        adj_cip = 0
        if product == 'A':
            adj_cip = cip * 0.6
        if product == 'B':
            adj_cip = cip
        return adj_cip

    # Predict the fuzzy inference given customer
    def expert_system_predict(row):
        for key in row.to_dict():
            if key not in ["index", "children"]:
                simulator.input[key] = row[key]
        simulator.compute()
        return simulator.output['cip']
    
    cust_predict = []
    # for index in tqdm(range(len(df))): 
    for index in tqdm(range(10)):
        row = df.loc[index]
        cip = expert_system_predict(row)
        product = model.predict(row)[0]
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

