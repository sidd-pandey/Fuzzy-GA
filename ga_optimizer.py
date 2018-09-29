import pandas as pd
import numpy as np

from pyevolve import G2DList, G1DList, Mutators
from pyevolve import GAllele, Initializators, GSimpleGA
from expert_system import ExpertSystem
from utils import load_data, expected_profit_campaign_predicted, expected_profit_campaign_actual
from utils import expected_profit_campaign_predicted_actual, matches_count
from prediction_system import PredictionModel

init_cutpoints = {
        "age": [30, 40, 50],
        "income": [2500, 5000, 7500],
        "avbal": [14000, 20000, 27000],
        "avtrans": [1000, 1500, 2400],
        "cip": [3, 5, 7]
    }

df = load_data("data/custdatabase.csv")

def convert_to_list(cutpoints):
    list_ = []
    for key, value in cutpoints.items():
        list_ += value
    return list_

def convert_to_dict(chromosome):
    dict_ = {}
    keys = init_cutpoints.keys()
    i = 0
    for key in keys:
        dict_[key] = chromosome[i:i+3]
        i += 3
    return dict_


chromosome_ = convert_to_list(init_cutpoints)

setOfAlleles = GAllele.GAlleles()
keys = init_cutpoints.keys()
for key in keys:
    for _ in range(0, 3):
        if key != 'cip':
            a = GAllele.GAlleleRange(int(df[key].min()), int(df[key].max()))
            setOfAlleles.add(a)
        else:
            a = GAllele.GAlleleRange(0, 10)
            setOfAlleles.add(a)

df_actual = pd.read_csv("data/Cust_Actual.csv")
df_actual_ = pd.read_csv("data/Cust_Actual.csv", index_col=['index'])

genome = G1DList.G1DList(len(chromosome_))
genome.setParams(allele=setOfAlleles)

def check_sorted(chromosome):
    i = 0
    for _ in range(5):
        original = chromosome[i:i+3]
        sorted_ = sorted(original)
        if (original != sorted_):
            return False
        i += 3
    return True

model = PredictionModel()

def eval_func(chromosome):
    cutpoints = convert_to_dict(chromosome)
    if check_sorted(chromosome):
        expert = ExpertSystem(df, cutpoints)
        expected_profit_campaign_pred, predicted_index = expected_profit_campaign_predicted(model, expert, df)
        expected_profit_campaign_act, actual_index = expected_profit_campaign_actual(df_actual)
        expected_profit_campaign_pred_act = expected_profit_campaign_predicted_actual(df_actual_, predicted_index)
        matches = matches_count(actual_index, predicted_index)
        return matches + 100*(abs(expected_profit_campaign_pred-expected_profit_campaign_act)) / expected_profit_campaign_act
#         print(matches)
#         return matches
    else:
        return 0
    
def t_init(genome, **args):
    genome.genomeList = chromosome_
    
#t_init(genome)

genome.evaluator.set(eval_func)
genome.mutator.set(Mutators.G1DListMutatorAllele)
genome.initializator.set(t_init)

# Genetic Algorithm Instance
ga = GSimpleGA.GSimpleGA(genome)
ga.setPopulationSize(100)
ga.setGenerations(500)

# Do the evolution, with stats dump
# frequency of 10 generations
ga.evolve(freq_stats=1)

# Best individual
print(ga.bestIndividual())