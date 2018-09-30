import pandas as pd
from tqdm import tqdm
import numpy as np

# Load and preprocess data from file
def load_data(file):
    df = pd.read_csv(file)
    if ("Unnamed: 10" in df.columns): df = df.drop("Unnamed: 10", axis =1)
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
    return df

# Expected profit for given customer
def expected_profit_customer(cip, product):
    adj_cip = 0
    if product == 'A':
        adj_cip = cip * 0.6
    if product == 'B':
        adj_cip = cip
    return adj_cip

# Compute expected profit for customers selected for campaign
def expected_profit_campaign_predicted(model, expert, df, save_csv=False):    
    cust_predict = []
    product_predict = model.predict(df)
    for index in tqdm(range(len(df))): 
        row = df.loc[index]
        cip = expert.predict(row)
        product = product_predict[index]
        expected_profit = expected_profit_customer(cip, product)
        cust_predict.append([df.loc[index, "index"], product, cip, expected_profit])   
    
    cust_predict_df = pd.DataFrame(cust_predict, columns=["index", "product", "cip", "expected profit"])
    cust_predict_df_sorted = cust_predict_df.sort_values(by=["expected profit"], ascending=False)
    cust_campaign_400 = cust_predict_df_sorted[:400]
    
    if save_csv:
        cust_predict_df.to_csv("data/Cust_Predict.csv", index=False)
        cust_campaign_400.to_csv("data/Cust_Predict_400.csv", index=False)

    expected_profit_campaign = np.sum(cust_campaign_400["expected profit"].values)

    return expected_profit_campaign, cust_campaign_400["index"].values

# Compute actual profit for customers selected for campaign
def expected_profit_campaign_actual(df):
    profit = []
    for index in range(len(df)):
        profit.append(expected_profit_customer(df.loc[index, "cust Investment Potential Score"], 
            df.loc[index, "status"]))
    
    df["profit"] = profit
    df_sorted = df.sort_values(by=["profit"], ascending=False)
    df_sorted_400 = df_sorted[:400]
    expected_profit = np.sum(df_sorted_400["profit"].values)
    
    return expected_profit, df_sorted_400["index"].values

# Compute actual profit corresponding to predicted indexes
def expected_profit_campaign_predicted_actual(df, indexs):
    profit = []
    for index in indexs:
        profit.append(expected_profit_customer(df.loc[index, "cust Investment Potential Score"], 
            df.loc[index, "status"]))
    
    expected_profit = np.sum(profit)

    return expected_profit

# Return the number of matches between actual and predicted
def matches_count(actual, predicted):
    matches = set(actual) & set(predicted)
return len(matches)