from expert_system import expert_system
import pandas as pd
from tqdm import tqdm
import numpy as np

# Read the data frame
df = pd.read_csv("data/custdatabase.csv").drop("Unnamed: 10", axis =1)

# Convert to Categorical
sex_map = {"M": 0, "F": 1}
mstatus_map = {"single":0, "married":1, "widowed":2, "divorced":3}
occupation_map = {'legal':0, 'IT':1, 'government':2, 'manuf':3, 'retired':4, 
                  'finance':5,'construct':6, 'education':7, 'medicine':8}
education_map = {'postgrad':3, 'secondary':0, 'tertiary':1, 'professional':2}
df["sex"] = df["sex"].map(sex_map)
df["mstatus"] = df["mstatus"].map(mstatus_map)
df["occupation"] = df["occupation"].map(occupation_map)
df["education"] = df["education"].map(education_map)  

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
    for index in tqdm(range(len(df))): 
        row = df.loc[index]
        cip = expert_system_predict(row)
        product = 'A'
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

