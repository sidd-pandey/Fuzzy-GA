# Fuzzy-GA
A Neuro GA Fuzzy Hybrid intelligent system


csv_prediction_model.py contains CsvPredictionModel which uses pre-saved prediction csv file as model. This prediction file is generated elsewhere. It contains prediction of their choice of product for 4000 customers, from custdatabase.csv file. This model exposes a predict function which takes in custdatabase.csv as df, uses it index and looks up in neural_network_pred_v1.csv, and return the corresponding value of decision field.


ga_optimizer.py script, executes genetic algorithm optimization to find better cut points for the membership function defined.
Run "python ga_optimizer.py" to start the script. At the end of execution, optimized cutpoints are prinited, which can be use base.py file to override the default cutpoints.

expert_system.py file contains class ExpertSystem which can be used to create instance of fuzzy inference system.

base.py combines the output from inference system and prediction model, to find expected profit and other metrics.
Run "python base.py" to start the script.

Line 13 loads the prediction model.
Line 14 loads an alternate prediction model which is RandomForestClassifier, uncomment this line to use this one.

Line 23 load the inference system with the cutpoints provided by dictionary ga_cutpoints at Line 15.
Line 24 load the inderence system with default cutpoints, uncomment this line to use default cutpoints.

