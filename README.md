# Sepsis-mortality-prediction
Costume code for mortality prediction 

The dataset used in this code (population_full.csv) is available from the corresponding author on reasonable request.
For more information regarding the variables included in the dataset, see supplemental file 1.

When running the code, follow these 3 steps:

Step 1 - Run mortalityPrediction with the csv-file (population_full.csv) including all variables to create a Balanced Random Forest to predict sepsis mortality, presenting the model´s accuracy and the mean feature importance for each variable included. The feature importance is then exported to a excel-file to be used in step 2.

Step 2 - Run exhaustiveSearch using excel-file (created in step 1) and the csv file (popilation_full.csv). This will create a total of 91 iteration of step 1 and for each iteration, the least important variable is removed from the training of the model. Plotting a graph presenting how the AUC and it´s standard deviation changes depending on how many variables are included in the training of the model.

Step 3 - Run mortalityPrediction again with a new csv-file, only including the most important variables showed in step 2 for finding the highest AUC.
