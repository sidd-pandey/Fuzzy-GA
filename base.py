import pandas as pd
from utils import load_data, expected_profit_campaign_predicted, expected_profit_campaign_actual
from utils import expected_profit_campaign_predicted_actual, matches_count
from prediction_system import PredictionModel
from expert_system import ExpertSystem

# Read the data frame
df = load_data("data/custdatabase.csv")

model = PredictionModel()
expert = ExpertSystem(df, cutpoints = {
        "age": [30, 40, 50],
        "income": [2500, 5000, 7500],
        "avbal": [14000, 20000, 27000],
        "avtrans": [1000, 1500, 2400],
        "cip": [3, 5, 7]
    })

expected_profit_campaign_pred, predicted_index = expected_profit_campaign_predicted(model, expert, df, save_csv=True)
print("Predicted:", expected_profit_campaign_pred)

df_actual = pd.read_csv("data/Cust_Actual.csv")
expected_profit_campaign_act, actual_index = expected_profit_campaign_actual(df_actual)
print("Actual:", expected_profit_campaign_act)

df_actual = pd.read_csv("data/Cust_Actual.csv", index_col=['index'])
expected_profit_campaign_pred_act = expected_profit_campaign_predicted_actual(df_actual, predicted_index)
print("Predicted Actual:", expected_profit_campaign_pred_act)

matches = matches_count(actual_index, predicted_index)
print("Number of Matches:", matches)