import pandas as pd
import numpy as np
import utils
import time
from pyevolve import G2DList, G1DList, Mutators
from pyevolve import GAllele, Initializators, GSimpleGA
from expert_system import ExpertSystem
from prediction_system import PredictionModel

init_cutpoints = {
        "age": [30, 40, 50],
        "income": [2500, 5000, 7500],
        "avbal": [14000, 20000, 27000],
        "avtrans": [1000, 1500, 2400],
        "cip": [3, 5, 7]
    }

df = utils.load_data("data/custdatabase.csv")
df = df.set_index("index")

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

df_actual = pd.read_csv("data/Cust_Actual.csv", index_col=['index'])

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
        subset = df.sample(50)
        expert = ExpertSystem(df, cutpoints)
        error = 1
        for index, row in subset.iterrows():
            cip_predicted = expert.predict(row) 
            cip_actual = df_actual.loc[index]["cust Investment Potential Score"]
            error += abs(cip_actual-cip_predicted)
        return 100/error;
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
ga.setGenerations(100)

# Do the evolution, with stats dump
# frequency of 10 generations
ga.evolve(freq_stats=1)

# Best individual
print(ga.bestIndividual())