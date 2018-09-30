# Fuzzy-GA
A Neuro GA Fuzzy Hybrid intelligent system


<i>csv_prediction_model.py</i> contains CsvPredictionModel which uses pre-saved prediction csv file as model. This prediction file is generated elsewhere. It contains prediction of their choice of product for 4000 customers, from custdatabase.csv file. This model exposes a predict function which takes in custdatabase.csv as df, uses it index and looks up in <i>neural_network_pred_v1.csv</i>, and return the corresponding value of decision field.<br>


<i>ga_optimizer.py</i> script, executes genetic algorithm optimization to find better cut points for the membership function defined.
Run <b>"python ga_optimizer.py"<b> to start the script. At the end of execution, optimized cutpoints are prinited, which can be use base.py file to override the default cutpoints.<br>

<i>expert_system.py</i> file contains class ExpertSystem which can be used to create instance of fuzzy inference system.<br>

<i>base.py</i> combines the output from inference system and prediction model, to find expected profit and other metrics.
Run <b>"python base.py"</b> to start the script.<br>

Line 13 loads the prediction model.<br>
Line 14 loads an alternate prediction model which is RandomForestClassifier, uncomment this line to use this one.<br>

Line 23 load the inference system with the cutpoints provided by dictionary ga_cutpoints at Line 15.<br>
Line 24 load the inderence system with default cutpoints, uncomment this line to use default cutpoints.<br>

