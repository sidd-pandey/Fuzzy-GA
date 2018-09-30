import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class ExpertSystem:

    init_cutpoints = {
        "age": [30, 40, 50],
        "income": [2500, 5000, 7500],
        "avbal": [14000, 20000, 27000],
        "avtrans": [1000, 1500, 2400],
        "cip": [3, 5, 7]
    }

    def __init__(self, df, cutpoints=None):
        self.df = df
        
        self.cutpoints = dict(self.init_cutpoints)
        if cutpoints is not None:
            for key in self.cutpoints:
                if key in cutpoints:
                    self.cutpoints[key] = cutpoints[key]
        
        self.simulator = self.expert_system()


    def expert_system(self):

        df = self.df
        cp = self.cutpoints
        # Membership function for sex
        # F = 1, M = 0
        x = np.arange(df["sex"].min(), df["sex"].max() + 1, 1)
        sex = ctrl.Antecedent(x, "sex")
        sex["M"] = np.array([1, 0])
        sex["F"] = np.array([0, 1])

        # Membership function for mstatus
        x = np.arange(df["mstatus"].min(), df["mstatus"].max()+1, 1)
        mstatus = ctrl.Antecedent(x, "mstatus")
        mstatus["single"] = np.array([1, 0, 0, 0])
        mstatus["married"] = np.array([0, 1, 0, 0])
        mstatus["widowed"] = np.array([0, 0, 1, 0])
        mstatus["divorced"] = np.array([0, 0, 0, 1])

        # Membership function for age
        # x = np.arange(df["age"].min(), df["age"].max()+1, 0.01)
        # x = sorted(df["age"])
        x = df["age"].sort_values().unique()
        age = ctrl.Antecedent(x, "age")
        age["young"] = fuzz.membership.trapmf(age.universe, [0, 0, cp["age"][0], cp["age"][1]])
        age["middle"] = fuzz.membership.trimf(age.universe, [cp["age"][0], cp["age"][1], cp["age"][2]])
        age["old"] = fuzz.membership.trapmf(age.universe, [cp["age"][1], cp["age"][2], max(x), max(x)])

        # Membership function for children
        x = np.arange(df["children"].min(), df["children"].max()+1, 1)
        children = ctrl.Antecedent(x, "children")
        children["low"] = np.array([1, 1, 0.5, 0, 0])
        children["high"] = np.array([0, 0, 0.5, 0.7, 1])


        # Membership function for occupation
        x = np.arange(df["occupation"].min(), df["occupation"].max()+1, 1)
        occupation = ctrl.Antecedent(x, "occupation")
        occupation["legal"] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        occupation["IT"] = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
        occupation["government"] = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
        occupation["manuf"] = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
        occupation["retired"] = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
        occupation["finance"] = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
        occupation["construct"] = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
        occupation["education"] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
        occupation["medicine"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

        # Membership function for education
        x = np.arange(df["education"].min(), df["education"].max()+1, 1)
        education = ctrl.Antecedent(x, "education")
        education["low"] = np.array([1, 1, 0, 0])
        education["high"] = np.array([0, 0, 1, 1])

        # Membership function for income
        # x = np.arange(df["income"].min(), df["income"].max()+1, 0.01)
        # x = sorted(df["income"])
        x = df["income"].sort_values().unique()
        income = ctrl.Antecedent(x, "income")
        income["low"] = fuzz.membership.trapmf(income.universe, [0, 0, cp["income"][0], cp["income"][1]])
        income["medium"] = fuzz.membership.trimf(income.universe, [cp["income"][0], cp["income"][1], cp["income"][2]])
        income["high"] = fuzz.membership.trapmf(income.universe, [cp["income"][1], cp["income"][2], max(x), max(x)])

        # Membership function for avbal
        # x = np.arange(df["avbal"].min(), df["avbal"].max()+1, 0.01)
        # x = sorted(df["avbal"])
        x = df["avbal"].sort_values().unique()
        avbal = ctrl.Antecedent(x, "avbal")
        avbal["low"] = fuzz.membership.trapmf(avbal.universe, [0, 0, cp["avbal"][0], cp["avbal"][1]])
        avbal["medium"] = fuzz.membership.trimf(avbal.universe, [cp["avbal"][0], cp["avbal"][1], cp["avbal"][2]])
        avbal["high"] = fuzz.membership.trapmf(avbal.universe, [cp["avbal"][1], cp["avbal"][2], max(x), max(x)])

        # Membership function for avtrans
        # x = np.arange(df["avtrans"].min(), df["avtrans"].max()+1, 0.01)
        # x = sorted(df["avtrans"])
        x = df["avtrans"].sort_values().unique()
        avtrans = ctrl.Antecedent(x, "avtrans")
        avtrans["low"] = fuzz.membership.trapmf(avtrans.universe, [0, 0, cp["avtrans"][0], cp["avtrans"][1]])
        avtrans["medium"] = fuzz.membership.trimf(avtrans.universe, [cp["avtrans"][0], cp["avtrans"][1], cp["avtrans"][2]])
        avtrans["high"] = fuzz.membership.trapmf(avtrans.universe, [cp["avtrans"][1], cp["avtrans"][2], max(x), max(x)])

        # Membership function for cip
        # x = np.arange(0, 10 + 1, 0.01)
        x = np.arange(0, 10 + 1, 0.1)
        cip = ctrl.Consequent(x, "cip")
        cip["low"] = fuzz.membership.trapmf(cip.universe, [0, 0, cp["cip"][0], cp["cip"][1]])
        cip["medium"] = fuzz.membership.trimf(cip.universe, [cp["cip"][0], cp["cip"][1], cp["cip"][2]])
        cip["high"] = fuzz.membership.trapmf(cip.universe, [cp["cip"][1], cp["cip"][2], max(x), max(x)])

        rules = []
    
        # Rules for Account Activity
        rules.append(ctrl.Rule(avbal["high"] & avtrans["high"], cip["high"]))
        rules.append(ctrl.Rule(avbal["high"] & avtrans["medium"], cip["medium"]))
        rules.append(ctrl.Rule(avbal["medium"] & avtrans["high"], cip["medium"]))
        rules.append(ctrl.Rule(avbal["medium"] & avtrans["medium"], cip["medium"]))
        rules.append(ctrl.Rule(avbal["low"] | avtrans["low"], cip["low"]))

        # Rules for Personal Factors
        rules.append(ctrl.Rule(sex["M"], cip["high"]))
        rules.append(ctrl.Rule(sex["F"] & mstatus["single"], cip["high"]))
        rules.append(ctrl.Rule(income["high"], cip["high"]))
        rules.append(ctrl.Rule(age["middle"], cip["high"]))
        rules.append(ctrl.Rule(occupation["retired"], cip["low"]))
        rules.append(ctrl.Rule(occupation["legal"] | occupation["medicine"] | occupation["education"]
                    | occupation["finance"] | occupation["IT"], cip["high"]))
        rules.append(ctrl.Rule(education["high"], cip["high"]))
        rules.append(ctrl.Rule(education["high"] & age["middle"], cip["high"]))
        rules.append(ctrl.Rule(income["high"] & age["old"], cip["high"]))

        # Rule Control System
        rule_ctrl = ctrl.ControlSystem(rules)
        simulator = ctrl.ControlSystemSimulation(rule_ctrl)

        return simulator

    def predict(self, row):
        def expert_system_predict(row):
            for key in row.to_dict():
                if key not in ["index", "children"]:
                    self.simulator.input[key] = row[key]
            self.simulator.compute()
            return self.simulator.output['cip']
        return expert_system_predict(row)


