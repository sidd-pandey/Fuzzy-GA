from expert_system import expert_system
import pandas as pd

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

def expected_profit_campaign(df):
    def expected_profit_customer(cip, product):
        adj_cip = 0
        if product == 'A':
            adj_cip = cip * 0.6
        if product == 'B':
            adj_cip = cip
        return adj_cip


cip = expert_system(1, df.loc[1], df)
print(cip)