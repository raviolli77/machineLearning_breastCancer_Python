# Machine Learning Techniques on Breast Cancer Wisconsin Data Set

**Contributor**:
+ Raul Eulogio

I created this repo as a way to get better acquainted with **Python** as a language and as a tool for data analysis. But it eventually became in exercise in utilizing various programming languages for machine learning applications.

I employed four **Machine Learning** techniques:
+ **Kth Nearest Neighbor**
+ **Random Forest**
+ **Neural Networks**:  

If you would like to see a walk through of the analysis on [inertia7](https://www.inertia7.com/projects/3) includes running code as well as explanations for exploratory analysis. This [project](https://www.inertia7.com/projects/95) contains an overview of *random forest*, explanations for other algorithms in the works.

The repository includes the *scripts* folder which contains scripts for these programming languages (in order of most detailed):
+ *Python*
+ *R*
+ *PySpark*

This repo is primarily concerned with the *python* iteration.

The multiple *python* script is broken into 5 sections (done by creating a script for each section) in the following order:
+ **Exploratory Analysis**
+ **Kth Nearest Neighbors**
+ **Random Forest**
+ **Neural Networks**
+ **Comparing Models**

**NOTE**: The files `data_extraction.py`, `helper_functions.py`, and `produce_model_metrics.py` are used to abstract functions to make code easier to read. These files do a lot of the work so if you are interested in how the scripts work definitely check them out.

## Running .py Script
A `virtualenv` is needed where you will download the necessary packages from the `requirements.txt` using:

	pip3 install -r requirements.txt

Once this is done you can run the scripts using the usual terminal command:

	$ python3 exploratory_analysis.py

**NOTE**: You can also run it by making script executable as such:

	$ chmod +x exploratory_analysis.py


**Remember**: You must have a *shebang* for this to run i.e. this must be at the very beginning of your script:

	#!/usr/bin/env python3

then you would simply just run it (I'll use **Random Forest** as an example)

	$ ./random_forest.py

## Conclusions
Once I employed all these methods, we were able to utilize various machine learning metrics. Each model provided valuable insight. *Kth Nearest Neighbor* helped create a baseline model to compare the more complex models. *Random forest* helps us see what variables were important in the bootstrapped decision trees. And *Neural Networks* provided minimal false negatives which results in false negatives. In this context it can mean death.  

### Diagnostics for Data Set


| Model/Algorithm      | Test Error Rate | False Negative for Test Set | Area under the Curve for ROC | Cross Validation Score        | Hyperparameter Optimization |
|----------------------|-----------------|-----------------------------|------------------------------|-------------------------------|-----------------------|
| Kth Nearest Neighbor | 0.07  | 5 | 0.980 | Accuracy:  0.925 (+/-  0.025) | Optimal *K* is 3 |
| Random Forest        | 0.035 | 3 | 0.996 | Accuracy:  0.963 (+/-  0.013) | {'max_features': 'log2', 'max_depth': 3, 'bootstrap': True, 'criterion': 'gini'}	|
| Neural Networks      | 0.035 | 1 | 0.982 | Accuracy:  0.967 (+/-  0.011) | {'hidden_layer_sizes': 12, 'activation': 'tanh', 'learning_rate_init': 0.05} |



#### ROC Curves for Data Set
<img src="https://raw.githubusercontent.com/raviolli77/machineLearning_breastCancer_Python/master/reports/images/roc_all.png" style="width: 100px;"/>

#### ROC Curves zoomed in
<img src="https://raw.githubusercontent.com/raviolli77/machineLearning_breastCancer_Python/master/reports/images/roc_all_zoom.png" style="width: 100px;"/>

The ROC Curves are more telling of **Random Forest** being a better model for predicting.

Any feedback is welcomed!

Things to do:
+ Create **Jupyter Notebook** for *KNN* and *NN* (1/25/2018)
+ Unit test scripts 
