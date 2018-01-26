# Table of Contents
+ [Introduction](#intro)
+ [Load Packages](#load_pack)
+ [Load Data](#load_data)
+ [Training and Test Sets](#train_test)
+ [Fitting Random Forest](#fit_model)
+ [Hyperparameters Optimization](#hype_opt)
+ [Out of Bag Error](#oob)
+ [Traditional Training and Test Split](#trad_train_test)
+ [Training Algorithm](#train_alg)
+ [Variable Importance](#var_imp)
+ [Cross Validation](#cross_val)
+ [Test Set Metrics](#test_set_met)
+ [ROC Curve Metrics](#roc_curve)
+ [Classification Report](#class_rep)
+ [Conclusions](#concl)

# <a name='intro'></a>Introduction

Random forests, also known as random decision forests, are a popular ensemble method that can be used to build predictive models for both classification and regression problems. Ensemble methods use multiple learning models to gain better predictive results — in the case of a random forest, the model creates an entire forest of random uncorrelated decision trees to arrive at the best possible answer.

To demonstrate how this works in practice — specifically in a classification context — I’ll be walking you through an example using a famous data set from the University of California, Irvine (UCI) Machine Learning Repository. The data set, called the Breast Cancer Wisconsin (Diagnostic) Data Set, deals with binary classification and includes features computed from digitized images of biopsies. The data set can be downloaded [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

To follow this tutorial, you will need some familiarity with classification and regression tree (CART) modeling. I will provide a brief overview of different CART methodologies that are relevant to random forest, beginning with decision trees. If you’d like to brush up on your knowledge of CART modeling before beginning the tutorial, I highly recommend reading Chapter 8 of the book “An Introduction to Statistical Learning with Applications in R,” which can be downloaded [here](http://www-bcf.usc.edu/~gareth/ISL/).

## Decision Trees

Decision trees are simple but intuitive models that utilize a top-down approach in which the root node creates binary splits until a certain criteria is met. This binary splitting of nodes provides a predicted value based on the interior nodes leading to the terminal (final) nodes. In a classification context, a decision tree will output a predicted target class for each terminal node produced.

Although intuitive, decision trees have limitations that prevent them from being useful in machine learning applications. You can learn more about implementing a decision tree [here](http://scikit-learn.org/stable/modules/tree.html).

### Limitations to Decision Trees

Decision trees tend to have high variance when they utilize different training and test sets of the same data, since they tend to overfit on training data. This leads to poor performance on unseen data. Unfortunately, this limits the usage of decision trees in predictive modeling. However, using ensemble methods, we can create models that utilize underlying decision trees as a foundation for producing powerful results.

## Bootstrap Aggregating Trees

Through a process known as bootstrap aggregating (or bagging), it’s possible to create an ensemble (forest) of trees where multiple training sets are generated with replacement, meaning data instances — or in the case of this tutorial, patients — can be repeated. Once the training sets are created, a CART model can be trained on each subsample.

This approach helps reduce variance by averaging the ensemble's results, creating a majority-votes model. Another important feature of bagging trees is that the resulting model uses the entire feature space when considering node splits. Bagging trees allow the trees to grow without pruning, reducing the tree-depth sizes and resulting in high variance but lower bias, which can help improve predictive power.

However, a downside to this process is that the utilization of the entire feature space creates a risk of correlation between trees, increasing bias in the model.

### Limitations to Bagging Trees

The main limitation of bagging trees is that it uses the entire feature space when creating splits in the trees. If some variables within the feature space are indicative of certain predictions, you run the risk of having a forest of correlated trees, thereby increasing bias and reducing variance.

However, a simple tweak of the bagging trees methodology can prove advantageous to the model’s predictive power.

## Random Forest

Random forest aims to reduce the previously mentioned correlation issue by choosing only a subsample of the feature space at each split. Essentially, it aims to make the trees de-correlated and prune the trees by setting a stopping criteria for node splits, which I will cover in more detail later.



# <a name="load_pack"></a>Load Packages

We load our modules into our python environment. In my case I am employing a `Jupyter Notebook` while running inside a `virtualenv` environment (the `requirements.txt` file associated with this repo contains the module information for reproducibility). 

We will be primarily using the [SciPy](https://www.scipy.org/stackspec.html) stack focusing on `pandas`, `matplotlib`, `seaborn` and `sklearn` for this tutorial. 


```python
# Import modules
%matplotlib inline

import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from urllib.request import urlopen 

plt.style.use('ggplot')
pd.set_option('display.max_columns', 500) 
```


# <a name='load_data'></a>Load Data
For this section, I'll load the data into a **Pandas** dataframe using `urlopen` from the `urllib.request` module. 

Instead of downloading a **csv**, I started implementing this method(Inspired by this [Python Tutorials](https://github.com/JasonFreeberg/PythonTutorials)) where I grab the data straight from the [UCI Machine Learning Database](https://archive.ics.uci.edu/ml/datasets.html) using an http request. 


**NOTE**: Original Data set can also be found [here](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)


```python
# Loading data and cleaning dataset
UCI_data_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases\
/breast-cancer-wisconsin/wdbc.data'
```

I do recommend on keeping a static file for your dataset as well.

Next, I created a list with the appropriate names and set them as the column names, once I load them unto a `pandas` data frame 


```python
names = ['id_number', 'diagnosis', 'radius_mean', 
         'texture_mean', 'perimeter_mean', 'area_mean', 
         'smoothness_mean', 'compactness_mean', 
         'concavity_mean','concave_points_mean', 
         'symmetry_mean', 'fractal_dimension_mean',
         'radius_se', 'texture_se', 'perimeter_se', 
         'area_se', 'smoothness_se', 'compactness_se', 
         'concavity_se', 'concave_points_se', 
         'symmetry_se', 'fractal_dimension_se', 
         'radius_worst', 'texture_worst', 
         'perimeter_worst', 'area_worst', 
         'smoothness_worst', 'compactness_worst', 
         'concavity_worst', 'concave_points_worst', 
         'symmetry_worst', 'fractal_dimension_worst'] 

dx = ['Benign', 'Malignant']
```


```python
breast_cancer = pd.read_csv(urlopen(UCI_data_URL), names=names)
```

## Cleaning
We do some minor cleanage like setting the `id_number` to be the data frame index, along with converting the `diagnosis` to the standard binary 1, 0 representation using the `map()` function. 


```python
# Setting 'id_number' as our index
breast_cancer.set_index(['id_number'], inplace = True) 
# Converted to binary to help later on with models and plots
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M':1, 'B':0})
```

## Missing Values
Given context of the data set, I know that there is no missing data, but I ran a `for-loop` that checks to see if there was any missing values through each column. Printing the column name and total missing values for that column, iteratively. 


```python
for col in breast_cancer:
    if ((breast_cancer[col].isnull().values.ravel().sum()) == 0):
        pass
    else:
        print(col)
        print((breast_cancer[col].isnull().values.ravel().sum()))
print('Sanity Check! \nNo missing Values found!')
```

    Sanity Check! 
    No missing Values found!


This will be used for the random forest model, where the `id_number` won't be relevant. 


```python
# For later use in CART models
names_index = names[2:]
```

Let's preview the data set utilizing the `head()` function which will give the first 5 values of our data frame. 


```python
breast_cancer.head()
```
<iframe width="798" height="280" frameborder="0" src='https://cdn.rawgit.com/raviolli77/7275dd3c9455f052171b22d3da5185b2/raw/fdc8c659fae9288f1ba1da01103265eb9403fc0b/head.html'></iframe>


Next, we'll give the dimensions of the data set; where the first value is the number of patients and the second value is the number of features. 

We print the data types of our data set this is important because this will often be an indicator of missing data, as well as giving us context to anymore data cleanage. 


```python
print("Here's the dimensions of our data frame:\n", 
     breast_cancer.shape)
print("Here's the data types of our columns:\n",
     breast_cancer.dtypes)
```

    Here's the dimensions of our data frame:
     (569, 31)
    Here's the data types of our columns:
     diagnosis                    int64
    radius_mean                float64
    texture_mean               float64
    perimeter_mean             float64
    area_mean                  float64
    smoothness_mean            float64
    compactness_mean           float64
    concavity_mean             float64
    concave_points_mean        float64
    symmetry_mean              float64
    fractal_dimension_mean     float64
    radius_se                  float64
    texture_se                 float64
    perimeter_se               float64
    area_se                    float64
    smoothness_se              float64
    compactness_se             float64
    concavity_se               float64
    concave_points_se          float64
    symmetry_se                float64
    fractal_dimension_se       float64
    radius_worst               float64
    texture_worst              float64
    perimeter_worst            float64
    area_worst                 float64
    smoothness_worst           float64
    compactness_worst          float64
    concavity_worst            float64
    concave_points_worst       float64
    symmetry_worst             float64
    fractal_dimension_worst    float64
    dtype: object


## Class Imbalance

The distribution for `diagnosis` is important because it brings up the discussion of *Class Imbalance* within Machine learning and data mining applications.

*Class Imbalance* refers to when a target class within a data set is outnumbered by the other target class (or classes). This can lead to misleading accuracy metrics, known as [accuracy paradox](https://en.wikipedia.org/wiki/Accuracy_paradox), therefore we have to make sure our target classes aren't imblanaced. 

We do so by creating a function that will output the distribution of the target classes. 

**NOTE**: If your data set suffers from class imbalance I suggest reading documentation on upsampling and downsampling.


```python
def print_dx_perc(data_frame, col):
    """Function used to print class distribution for our data set"""
    dx_vals = data_frame[col].value_counts()
    dx_vals = dx_vals.reset_index()
    # Create a function to output the percentage 
    f = lambda x, y: 100 * (x / sum(y))
    for i in range(0, len(dx)):
        print('{0} accounts for {1:.2f}% of the diagnosis class'\
              .format(dx[i], f(dx_vals[col].iloc[i], 
                               dx_vals[col])))
```


```python
print_dx_perc(breast_cancer, 'diagnosis')
```

    Benign accounts for 62.74% of the diagnosis class
    Malignant accounts for 37.26% of the diagnosis class


Fortunatly, this data set does not suffer from *class imbalance*. 

Next we will use a useful function that gives us standard descriptive statistics for each feature including mean, standard deviation, minimum value, maximum value, and range intervals. 


```python
breast_cancer.describe()
```

<iframe width="798" height="350" frameborder="0" src='https://cdn.rawgit.com/raviolli77/f0b743ab560331bf9382d9a6ad0b8ff7/raw/54d0f327ed07b42294da40303380530edd7b9f3d/describe_breast_cancer.html'></iframe>


We can see through the maximum row that our data varies in distribution, this will be important when considering classification models. 

*Standardization* is an important requirement for many classification models that should be considered when implementing pre-processing. Some models (like neural networks) can perform poorly if pre-processing isn't considered, so the `describe()` function can be a good indicator for *standardization*. Fortunately Random Forest does not require any pre-processing (for use of categorical data see [sklearn's Encoding Categorical Data](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features) section).


# <a name='train_test'></a>Creating Training and Test Sets

We split the data set into our *training* and *test sets* which will be (pseudo) randomly selected having a 80-20% splt. We will use the training set to train our model along with some optimization, and use our test set as the unseen data that will be a useful final metric to let us know how well our model does. 

When using this method for machine learning always be weary of utilizing your test set when creating models. The issue of data leakage is a grave and serious issue that is common in practice and can result in over-fitting. More on data leakage can be found in this [Kaggle article](https://www.kaggle.com/wiki/Leakage)


```python
feature_space = breast_cancer.iloc[:, breast_cancer.columns != 'diagnosis']
feature_class = breast_cancer.iloc[:, breast_cancer.columns == 'diagnosis']


training_set, test_set, \
class_set, test_class_set = train_test_split(feature_space,
											feature_class,
                                            test_size = 0.20, 
                                            random_state = 42)
```

**NOTE**: What I mean when I say *pseudo-random* is that we would want everyone who replicates this project to get the same results. So we use a random seed generator and set it equal to a number of our choosing, this will then make the results the same for anyone who uses this generator, awesome for reproducibility.


```python
# Cleaning test sets to avoid future warning messages
class_set = class_set.values.ravel() 
test_class_set = test_class_set.values.ravel() 
```


# <a name='fit_model'></a>Fitting Random Forest

Now we will create the model stating no parameter tuning aside from the random seed generator. 

What I mean when I say parameter tuning is different machine learning models utilize various parameters which have to be tuned by the person implementing the algorithm. Here I'll give a brief overview of the parameters I will be tuning in this tutorial:

+ max_depth: the maximum splits for all trees in the forest. 
+ bootstrap: indicating whether or not we want to use bootstrap samples when building trees
+ max_features: the maximum number of features that will be used in the node splitting (the main difference previously mentioned between Bagging trees and Random Forest). Typically we want a value that is less than p, where p is all features in our dataset. 
+ criterion: this is the metric used to asses the stopping criteria for the Decision trees, more on this later 


```python
# Set the random state for reproducibility
fit_rf = RandomForestClassifier(random_state=42)
```

# <a name='hype_opt'></a>Hyperparameters Optimization 

Utilizing the `GridSearchCV` functionality, I create a dictionary with parameters I am looking to optimize to create the best model for our data. Setting the `n_jobs` to 3 tells the grid search to run 3 jobs in parallel reducing the time the function will take to compute the best parameters. I included the timer to help see how long different jobs took, ultimately deciding on using 3. 


This will help set parameters which I will then use to tune one more paramter; the number of trees. 


```python
np.random.seed(42)
start = time.time()

param_dist = {'max_depth': [2, 3, 4],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']}

cv_rf = GridSearchCV(fit_rf, cv = 10,
                     param_grid=param_dist, 
                     n_jobs = 3)

cv_rf.fit(training_set, class_set)
print('Best Parameters using grid search: \n', 
      cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))
```

    Best Parameters using grid search: 
     {'max_features': 'log2', 'criterion': 'gini', 'bootstrap': True, 'max_depth': 3}
    Time taken in grid search:  9.71


Once we are given the best parameters, we set the parameters to our model. 

Notice how we didn't utilize the `bootstrap: True` parameter, this will be because of the following section. 


```python
# Set best parameters given by grid search 
fit_rf.set_params(criterion = 'gini',
                  max_features = 'log2', 
                  max_depth = 3)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=3, max_features='log2', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=10, n_jobs=1, oob_score=False, random_state=42,
                verbose=0, warm_start=False)

# <a name='oob'></a>Out of Bag Error Rate

Another useful feature of Random Forest is the concept of Out of Bag Error Rate or OOB error rate. When creating the forest, typically only 2/3 of the data is used to train the trees, this gives us 1/3 of unseen data that we can then utilize in a way that is advantageos to our accuracy metrics withou being computationally expensive like cross validation. 

When calculating OOB, two parameters have to be changed as outlined below. Also utilizing a `for-loop` across a multitude of forest sizes, we can calculate the OOB Error rate and use this to asses how many trees are appropriate for our model!


```python
fit_rf.set_params(warm_start=True, 
                  oob_score=True)

min_estimators = 15
max_estimators = 1000

error_rate = {}

for i in range(min_estimators, max_estimators + 1):
    fit_rf.set_params(n_estimators=i)
    fit_rf.fit(training_set, class_set)

    oob_error = 1 - fit_rf.oob_score_
    error_rate[i] = oob_error
```


```python
# Convert dictionary to a pandas series for easy plotting 
oob_series = pd.Series(error_rate)
```


```python
fig, ax = plt.subplots(figsize=(10, 10))

ax.set_axis_bgcolor('#fafafa')

oob_series.plot(kind='line',
                color = 'red')
plt.axhline(0.055, 
            color='#875FDB',
           linestyle='--')
plt.axhline(0.05, 
            color='#875FDB',
           linestyle='--')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error')
plt.title('OOB Error Across Trees \n(From 15 to 1000 trees)')
```


<img src='https://raw.githubusercontent.com/raviolli77/machineLearning_breastCancer_Python/master/notebooks/random_forest_files/random_forest_36_1.png'>


The OOB error starts to oscilate at around 400 trees, so I will go ahead and use my judgement to use 400 trees in my forest. Using the `pandas` series object I can easily find the OOB error rate for the estimator as follows:


```python
print('OOB Error rate for 400 trees is: {0:.5f}'.format(oob_series[400]))
```

    OOB Error rate for 400 trees is: 0.04835


Utilizing the OOB error rate that was created gives us an unbiased error rate. This can be helpful when cross validating and/or hyperparameter optimization prove to be too computationally expensive. 

For the sake of this tutorial I will go over the other traditional methods for machine learning including the training and test error route, along with cross validation metrics.

# <a name='trad_train_test'></a>Traditional Training and Test Set Split

In order for this methodology to work we will set the number of trees calculated using the OOB error rate, and removing the `warm_start` and `oob_score` parameters. Along with including the `bootstrap` parameter. 


```python
fit_rf.set_params(n_estimators=400,
                  bootstrap = True,
                  warm_start=False, 
                  oob_score=False)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=3, max_features='log2', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=400, n_jobs=1, oob_score=False, random_state=42,
                verbose=0, warm_start=False)


# <a name='train_alg'></a>Training Algorithm

Next we train the algorithm utilizing the training and target class set we had made earlier. 


```python
fit_rf.fit(training_set, class_set)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=3, max_features='log2', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=400, n_jobs=1, oob_score=False, random_state=42,
                verbose=0, warm_start=False)


# <a name='var_imp'></a>Variable Importance

Once we have trained the model, we are able to assess this concept of variable importance. A downside to creating ensemble methods with Decision Trees is we lose the interpretability that a single tree gives. A single tree can outline for us important node splits along with variables that were important at each split. 


Forunately ensemble methods utilzing CART models use a metric to evaluate homogeneity of splits. Thus when creating ensembles these metrics can be utilized to give insight to important variables used in the training of the model. Two metrics that are used are `gini impurity` and `entropy`. 

The two metrics vary and from reading documentation online, many people favor `gini impurity` due to the computational cost of `entropy` since it requires calculating the logarithmic function. For more discussion I recommend reading this [article](https://github.com/rasbt/python-machine-learning-book/blob/master/faq/decision-tree-binary.md).

Here we define each metric:

$$Gini\ Impurity = 1 - \sum_i p_i$$

$$Entropy = \sum_i -p_i * \log_2 p_i$$

where $p_i$ is defined as the proportion of subsamples that belong to a certain target class. 

We are able to access the feature importance of the model and using a helper function output the importance of our variables in descending order. 


```python
importances_rf = fit_rf.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]
```


```python
def variable_importance(importance, indices):
    """
    Purpose:
    ----------
    Prints dependent variable names ordered from largest to smallest
    based on gini or information gain for CART model. 
    
    Parameters:
    ----------
    names:      Name of columns included in model
    importance: Array returned from feature_importances_ for CART
                   models organized by dataframe index
    indices:    Organized index of dataframe from largest to smallest
                   based on feature_importances_

    Returns:
    ----------
    Print statement outputting variable importance in descending order
    """
    print("Feature ranking:")
    
    for f in range(len(names_index)):
        i = f
        print("%d. The feature '%s' \
has a Mean Decrease in Gini of %f" % (f + 1, 
                                         names_index[indices[i]], 
                                         importance[indices[f]]))
```


```python
variable_importance(importances_rf, indices_rf)
```

    Feature ranking:
    1. The feature 'area_worst' has a Mean Decrease in Gini of 0.129856
    2. The feature 'perimeter_worst' has a Mean Decrease in Gini of 0.120953
    3. The feature 'concave_points_worst' has a Mean Decrease in Gini of 0.115548
    4. The feature 'concave_points_mean' has a Mean Decrease in Gini of 0.100136
    5. The feature 'radius_worst' has a Mean Decrease in Gini of 0.078047
    6. The feature 'concavity_mean' has a Mean Decrease in Gini of 0.062143
    7. The feature 'area_mean' has a Mean Decrease in Gini of 0.056556
    8. The feature 'radius_mean' has a Mean Decrease in Gini of 0.054567
    9. The feature 'perimeter_mean' has a Mean Decrease in Gini of 0.051745
    10. The feature 'area_se' has a Mean Decrease in Gini of 0.043261
    11. The feature 'concavity_worst' has a Mean Decrease in Gini of 0.038659
    12. The feature 'compactness_worst' has a Mean Decrease in Gini of 0.020329
    13. The feature 'compactness_mean' has a Mean Decrease in Gini of 0.016163
    14. The feature 'texture_worst' has a Mean Decrease in Gini of 0.015542
    15. The feature 'radius_se' has a Mean Decrease in Gini of 0.014521
    16. The feature 'perimeter_se' has a Mean Decrease in Gini of 0.013084
    17. The feature 'texture_mean' has a Mean Decrease in Gini of 0.012203
    18. The feature 'symmetry_worst' has a Mean Decrease in Gini of 0.011750
    19. The feature 'smoothness_worst' has a Mean Decrease in Gini of 0.009380
    20. The feature 'concavity_se' has a Mean Decrease in Gini of 0.009105
    21. The feature 'concave_points_se' has a Mean Decrease in Gini of 0.004449
    22. The feature 'smoothness_mean' has a Mean Decrease in Gini of 0.003982
    23. The feature 'fractal_dimension_se' has a Mean Decrease in Gini of 0.003953
    24. The feature 'fractal_dimension_worst' has a Mean Decrease in Gini of 0.002672
    25. The feature 'fractal_dimension_mean' has a Mean Decrease in Gini of 0.002210
    26. The feature 'smoothness_se' has a Mean Decrease in Gini of 0.002169
    27. The feature 'symmetry_mean' has a Mean Decrease in Gini of 0.002051
    28. The feature 'texture_se' has a Mean Decrease in Gini of 0.002043
    29. The feature 'symmetry_se' has a Mean Decrease in Gini of 0.001937
    30. The feature 'compactness_se' has a Mean Decrease in Gini of 0.000987


We can see here that our top 5 variables were `area_worst`, `perimeter_worst`, `concave_points_worst`, `concave_points_mean`, `radius_worst`. 

This can give us great insight for further analysis like [feature engineering](https://en.wikipedia.org/wiki/Feature_engineering), although we won't go into this during this tutorial. This step can help give insight to the practitioner and audience as to what variables played an important part to the predictions generated by the model. 

In our test case, this can help people in the medical field focus on the top variables and their relationship with breast cancer. 


```python
def variable_importance_plot(importance, indices):
    """
    Purpose
    ----------
    Prints bar chart detailing variable importance for CART model 
    NOTE: feature_space list was created because the bar chart 
    was transposed and index would be in incorrect order.

    Parameters
    ----------
    importance_desc: Array returned from feature_importances_ for CART
                        models organized in descending order 

    indices: Organized index of dataframe from largest to smallest
                        based on feature_importances_ 
    Returns:
    ----------
    Returns variable importance plot in descending order
    """
    index = np.arange(len(names_index))
    
    importance_desc = sorted(importance) 
    feature_space = []
    for i in range(29, -1, -1):
        feature_space.append(names_index[indices[i]])

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_axis_bgcolor('#fafafa')
    plt.title('Feature importances for Random Forest Model\
    \nBreast Cancer (Diagnostic)')
    plt.barh(index, 
             importance_desc,
             align="center", 
             color = '#875FDB')
    plt.yticks(index, 
               feature_space)

    plt.ylim(-1, 30)
    plt.xlim(0, max(importance_desc))
    plt.xlabel('Mean Decrease in Gini')
    plt.ylabel('Feature')
    
    plt.show()
    plt.close()
```


```python
variable_importance_plot(importances_rf, indices_rf)
```


<img src='https://raw.githubusercontent.com/raviolli77/machineLearning_breastCancer_Python/master/notebooks/random_forest_files/random_forest_49_0.png'>

The visual helps drive the point of variable importance, since you can clearly see the difference in importance of variables for the ensemble method. Certain cutoff points can be made to reduce the inclusion of features and can help in the accuracy of the model, since we'll be removing what is considered noise within our feature space.

# <a name='cross_val'></a>Cross Validation

*Cross validation* is a powerful tool that is used for estimating the predicitive power of your model, which performs better than the conventional training and test set. What we are doing with *Cross Validation* is we are essentially creating multiple training and test sets, then averaging the scores to give us a less biased metric.

In our case we are creating 10 sets within our data set that calculates the estimations we have done already, but then averages the prediction error to give us a more accurate representation of our model's prediction power, since the model's performance can vary significantly when utilizing different training and test sets.

**Suggested Reading**: For a more concise explanation of *Cross Validation* I recommend reading [An Introduction to Statistical Learnings with Applications in R](http://www-bcf.usc.edu/~gareth/ISL/), specifically chapter 5.1!

## K-Fold Cross Validation

Here we are employing *K-Fold Cross Validation*, more specifically 10 folds. So therefore we are creating 10 subsets of our data where we will be employing the training and test set methodology then averaging the accuracy for all folds to give us our estimatation.

Within a Random Forest context if your data set is significantly large one can choose to not do cross validation and use the OOB error rate as an unbiased metric for computational costs, but for this tutorial I included it to show different accuracy metrics available. 


```python
def cross_val_metrics(fit, training_set, class_set, print_results = True):
    """
    Purpose
    ----------
    Function helps automate cross validation processes while including 
    option to print metrics or store in variable
    
    Parameters
    ----------
    fit: Fitted model 
    training_set:  Data_frame containing 80% of original dataframe
    class_set:     data_frame containing the respective target vaues 
                      for the training_set
    print_results: Boolean, if true prints the metrics, else saves metrics as 
                      variables

    Returnss
    ----------
    scores.mean(): Float representing cross validation score
    scores.std() / 2: Float representing the standard error (derived
                from cross validation score's standard deviation)
    """
    n = KFold(n_splits=10)
    scores = cross_val_score(fit, 
                         training_set, 
                         class_set, 
                         cv = n)
    if print_results:
        print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
              .format(scores.mean(), scores.std() / 2))
    else:
        return scores.mean(), scores.std() / 2
```


```python
cross_val_metrics(fit_rf, 
                  training_set, 
                  class_set, 
                  print_results = True)
```

    Accuracy:  0.947 (+/-  0.019)


# <a name='test_set_met'></a>Test Set Metrics
Now we will be utilizing the test set that was created earlier to receive another metric for evaluation of our model. Recall the importance of data leakage and that we didn't touch the test set until now, after we had done hyperparamter optimization. 

We create a confusion matrix showcasing the following metrics:

| n = Sample Size | Predicted Benign | Predicted Malignant | 
|-----------------|------------------|---------------------|
| Actual Benign | *True Positive* | *False Negative* | 
| Actual Malignant | *False Positive* | *True Negative* | 


```python
predictions_rf = fit_rf.predict(test_set)
```


```python
test_crosstb = pd.crosstab(index = test_class_set,
                           columns = predictions_rf)

# More human readable 
test_crosstb = test_crosstb.rename(columns= {0: 'Benign', 1: 'Malignant'})
test_crosstb.index = ['Benign', 'Malignant']
test_crosstb.columns.name = 'n = 114'
```


```python
test_crosstb
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>n = 114</th>
      <th>Benign</th>
      <th>Malignant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Benign</th>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Malignant</th>
      <td>3</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
accuracy_rf = fit_rf.score(test_set, test_class_set)

print("Here is our mean accuracy on the test set:\n {0:.3f}"\
      .format(accuracy_rf))
```

    Here is our mean accuracy on the test set:
     0.965



```python
# Here we calculate the test error rate!
test_error_rate_rf = 1 - accuracy_rf
print("The test error rate for our model is:\n {0: .4f}"\
      .format(test_error_rate_rf))
```

    The test error rate for our model is:
      0.0351


As you can see we got a very similar error rate for our test set that we did for our OOB, which is a good sign for our model. 


# <a name='roc_curve'></a>ROC Curve Metrics

Receiver Operating Characteristc Curve, calculates the False Positive Rates and True Positive Rates across different thresholds . 

We will now graph these calculations, and being located the top left corner of the plot indicates a really ideal model, i.e. a False Positive Rate of 0 and True Positive Rate of 1, whereas an ROC curve that is at a 45 degree is indicative of a model that is essentially randomly guessing. 

We also calculated the Area under the Curve or AUC, the AUC is used as a metric to differentiate the predicion power for those with the disease and those without it. Typically a value closer to one means that our model was able to differentiate correctly from a random sample of the two target classes of two patients which had and which didn't have the disease.    


```python
fpr2, tpr2, _ = roc_curve(predictions_rf, 
                          test_class_set)
```


```python
auc_rf = auc(fpr2, tpr2)
```


```python
def plot_roc_curve(fpr, tpr, auc, mod, xlim=None, ylim=None):
    """
    Purpose
    ----------
    Function creates ROC Curve for respective model given selected parameters.
    Optional x and y limits to zoom into graph 
    
    Parameters
    ----------
    fpr:  Array returned from sklearn.metrics.roc_curve for increasing 
             false positive rates
    tpr:  Array returned from sklearn.metrics.roc_curve for increasing 
             true positive rates
    auc:  Float returned from sklearn.metrics.auc (Area under Curve)
    mod:  String represenation of appropriate model, can only contain the 
             following: ['knn', 'rf', 'nn']
    xlim: Set upper and lower x-limits
    ylim: Set upper and lower y-limits
    
    Returns:
    ----------
    Returns plot of Receiving Operating Curve for specific model. Allowing user to 
    specify x and y-limits. 
    """
    mod_list = ['knn', 'rf', 'nn']
    method = [('Kth Nearest Neighbor', 'deeppink'), 
              ('Random Forest', 'red'), 
              ('Neural Network', 'purple')]

    plot_title = ''
    color_value = ''
    for i in range(0, 3):
        if mod_list[i] == mod:
            plot_title = method[i][0]
            color_value = method[i][1]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_bgcolor('#fafafa')

    plt.plot(fpr, tpr, 
             color=color_value, 
             linewidth=1)
    plt.title('ROC Curve For {0} (AUC = {1: 0.3f}) \
              \nBreast Cancer Diagnostic'\
              .format(plot_title, auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
    plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
    plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.close()
```


```python
plot_roc_curve(fpr2, tpr2, auc_rf, 'rf',
               xlim=(-0.01, 1.05), 
               ylim=(0.001, 1.05))
```


<img src='https://raw.githubusercontent.com/raviolli77/machineLearning_breastCancer_Python/master/notebooks/random_forest_files/random_forest_63_0.png'>

Our model did exceptional with an AUC over .90, now we do a zoomed in view to showcase the closeness our ROC Curve is relative to the ideal ROC Curve. 


```python
plot_roc_curve(fpr2, tpr2, auc_rf, 'rf', 
               xlim=(-0.01, 0.2), 
               ylim=(0.85, 1.01))
```


<img src='https://raw.githubusercontent.com/raviolli77/machineLearning_breastCancer_Python/master/notebooks/random_forest_files/random_forest_65_0.png'>


# <a name='class_rep'></a>Classification Report

The classification report is available through `sklearn.metrics`, this report gives many important classification metrics including:
+ `Precision`: also the positive predictive value, is the number of correct predictions divided by the number of correct predictions plus false positives, so $tp / (tp + fp)$
+ `Recall`: also known as the sensitivity, is the number of correct predictions divided by the total number of instances so $tp / (tp + fn)$ where $fn$ is the number of false negatives
+ `f1-score`: this is defined as the *weighted harmonic mean* of both the precision and recall, where the f1-score at 1 is the best value and worst value at 0, as defined by the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support)
+ `support`: number of instances that are the correct target values

Across the board we can see that our model provided great insight into classifying patients based on the FNA scans. Important metrics to consider would be optimzing the *false positive* rate since within this context it would be bad for the model to tell someone that they are cancer free when in reality they have cancer.



```python
def print_class_report(predictions, alg_name):
    """
    Purpose
    ----------
    Function helps automate the report generated by the
    sklearn package. Useful for multiple model comparison

    Parameters:
    ----------
    predictions: The predictions made by the algorithm used
    alg_name: String containing the name of the algorithm used
    
    Returns:
    ----------
    Returns classification report generated from sklearn. 
    """
    print('Classification Report for {0}:'.format(alg_name))
    print(classification_report(predictions, 
            test_class_set, 
            target_names = dx))
```


```python
class_report = print_class_report(predictions_rf, 'Random Forest')
```

    Classification Report for Random Forest:
                 precision    recall  f1-score   support
    
         Benign       0.99      0.96      0.97        73
      Malignant       0.93      0.98      0.95        41
    
    avg / total       0.97      0.96      0.97       114
    


## Metrics for Random Forest

Here I've accumulated the various metrics we used through this tutorial in a simple table! For the original analysis I compared *Kth Nearest Neighbor*, *Random Forest*, and *Neural Networks*, so most of the analysis was done to compare across different models. 

| Model | OOB Error Rate | Test Error Rate | Cross Validation Score | AUC | 
|-------|----------------|------------------------|-----------------|-----|
| Random Forest | 0.04835 |  0.0351 | 0.947 (+/-  0.019) | 0.967 | 



# <a name='concl'></a>Conclusions

For this tutorial we went through a number of metrics to assess the capabilites of our Random Forest, but this can be taken further when using background information of the data set. Feature engineering would be a powerful tool to extract and move forward into research regarding the important features. As well defining key metrics to utilize when optimizing model paramters. 

There have been advancements with image classification in the past decade that utilize the images intead of extracted features from images, but this data set is a great resource to become with machine learning processes. Especially for those who are just beginning to learn machine learning concepts. If you have any suggestions, recommendations, or corrections please reach out to me.