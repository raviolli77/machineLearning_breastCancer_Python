# Table of Contents
+ [Setting Up Python Environment](#pyenv)
+ [Loading Data](#loaddata)
+ [Exploratory Analysis](#exploranal)
+ [Visual Exploratory Analysis](#visexploranal)
+ [Model Estimation](#modelest) 
+ [Kth Nearest Neighbor](#knn) 
+ [Random Forest](#randforest)
+ [Neural Networks](#neurnetwork)
+ [ROC Curves](#roccurve)
+ [Conclusions](#conclusions)

# Load Modules <a name='pyenv'></a>
We load our modules into our python environment. In my case I am employing a **Jupyter Notebook** while running inside a **virtualenv** environment. 


```python
%matplotlib inline

import numpy as np
import pandas as pd # Data frames
import matplotlib.pyplot as plt # Visuals
import seaborn as sns # Danker visuals
from helperFunctions import *
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold, cross_val_score 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import export_graphviz 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # Calculating AUC for ROC's!
from urllib.request import urlopen 

pd.set_option('display.max_columns', 500) 
# Included to show all the columns 
# since it is a fairly large data set

plt.style.use('ggplot') # Using ggplot2 style visuals 
# because that's how I learned my visuals 
# and I'm sticking to it!
```

# Loading Data <a name='loaddata'></a>
For this section, I'll load the data into a **Pandas** dataframe using `urlopen` from the `urllib.request` module. 

Instead of downloading a **csv**, I started implementing this method(Inspired by [Jason's Python Tutorials](https://github.com/JasonFreeberg/PythonTutorials)) where I grab the data straight from the [UCI Machine Learning Database](https://archive.ics.uci.edu/ml/datasets.html). Makes it easier to go about analysis from online sources and cuts out the need to download/upload a **csv** file when uploading on *GitHub*. I created a list with the appropriate names and set them within the dataframe. 

**NOTE**: The names were not documented to well so I used [this analysis](https://www.kaggle.com/buddhiniw/d/uciml/breast-cancer-wisconsin-data/breast-cancer-prediction) (I will refer to it as *Buddhini W.* from now on) to grab the variable names and some other tricks that I didn't know that were available in *Python* (I will mention the use in the script!)

Finally I set the column `id_number` as the index for the dataframe. 


```python
UCI_data_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

names = ['id_number', 'diagnosis', 'radius_mean', 
         'texture_mean', 'perimeter_mean', 'area_mean', 
         'smoothness_mean', 'compactness_mean', 'concavity_mean',
         'concave_points_mean', 'symmetry_mean', 
         'fractal_dimension_mean', 'radius_se', 'texture_se', 
         'perimeter_se', 'area_se', 'smoothness_se', 
         'compactness_se', 'concavity_se', 'concave_points_se', 
         'symmetry_se', 'fractal_dimension_se', 
         'radius_worst', 'texture_worst', 'perimeter_worst',
         'area_worst', 'smoothness_worst', 
         'compactness_worst', 'concavity_worst', 
         'concave_points_worst', 'symmetry_worst', 
         'fractal_dimension_worst'] 

breastCancer = pd.read_csv(urlopen(UCI_data_URL), names=names)

# Setting 'id_number' as our index
breastCancer.set_index(['id_number'], inplace = True) 
namesInd = names[2:] # FOR CART MODELS LATER
```

# Exploratory Analysis <a name='exploranal'></a>

An important process in **Machine Learning** is doing **Exploratory Analysis** to get a *feel* for your data. As well as creating visuals that can be digestable for anyone of any skill level. Many people will try to jump into the predictive modeling, but if you don't know how your data interacts you're really doing yourself an injustice when choosing and justifying models. 

Its good to always output sections of your data so you can give context to the reader as to what each column looks like, as well as seeing examples of how the data is suppose to be formatted when loaded correctly. Many people run into the issue (especially if you run into a data set with poor documentation w.r.t. the column names), so its good habit to show your data during your analysis. 

We use the function `head()` which is essentially the same as the `head` function in *R* if you come from an *R* background. Notice the syntax for *Python* is significantly different than that of *R* though. 


```python
breastCancer.head()
```
<!-- Missing output -->

### More Preliminary Analysis
Much of these sections are given to give someone context to the dataset you are utilizing. Often looking at raw data will not give people the desired context, so it is important for us as data enthusiast to fill in the gaps for people who are interested in the analysis. But don't plan on running it anytime soon. 

#### Data Frame Dimensions
Here we use the `.shape` function to give us the lengths of our data frame, where the first output is the row-length and the second output is the column-length. 

#### Data Types
Another piece of information that is **important** is the data types of your variables in the data set. 

It is often good practice to check the variable types with either the source or with your own knowledge of the data set. For this data set, we know that all variables are measurements, so they are all continous (Except **Dx**), so no further processing is needed for this step.

An common error that can happen say a variable is *discrete* (or *categorical*), but has a numerical representation someone can easily forget the pre-processing and do analysis on the data type as is. Since they are numeric they will be interpretted as either `int` or `float` (more importantly as continous) which is incorrect. I can go on and on, but for this data set the numerical representation of the **Dx** is correct and is referred to as *indicator* or *dummy* variables. 


```python
print("Here's the dimensions of our data frame:\n", 
     breastCancer.shape)
print("Here's the data types of our columns:\n",
     breastCancer.dtypes)
```
### Terminal Output 

    Here's the dimensions of our data frame:
     (569, 31)
    Here's the data types of our columns:
     diagnosis                   object
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


As you can see we'll be dealing mostly with `float` types! For our analysis our next step is converting the Diagnoses into the appropriate binary representation.

## Converting Diagnoses
Important when doing analysis, converting variable types to the appropriate representation. A tool is as useful as the person utilizing it, so if we enter our data incorrectly the algorithm will suffer not as a result from its capabilities, but from the human component (More on this later). 

Here I converted the Dx to **binary** represenations using the `map` functionality in `pandas`. I borrowed this from *Buddhini W*. We are using a dictionary to map out this conversion:

    {'M':1, 'B':0}

which then converts the previous string representations of the Dx to the **binary** representation, where 1 == **Malignant** and 0 == **Benign**. 


```python
# Converted to binary to help later on with models and plots
breastCancer['diagnosis'] = breastCancer['diagnosis']\
  .map({'M':1, 'B':0})

# Let's look at the count of the new representations of our Dx's
breastCancer['diagnosis'].value_counts()
```
### Terminal Output 



    0    357
    1    212
    Name: diagnosis, dtype: int64



## Class Imbalance
The count for our Dx is important because it brings up the discussion of *Class Imbalance* within *Machine learning* and *data mining* applications. 

*Class Imbalance* refers to when a class within a data set is outnumbered by the other class (or classes). 
Reading documentation online, *Class Imbalance* is present when a class populates 10-20% of the data set. 

However for this data set, its pretty obvious that we don't suffer from this, but since I'm practicing my **Python**, I decided to experiment with *functions* to get better at **Python**! 
<br>

**NOTE**: If your data set suffers from *class imbalance* I suggest reading documentation on *upsampling* and *downsampling*. 

```python
def classImbalance(item):
    '''
    Goal of this function:
    Loops through the Dx to print percentage of class distributions 
    w.r.t. the length of the data set
    '''
    i = 0
    n = 0
    perMal = 0 
    perBeg = 0
    for item in breastCancer[item]:
        if (item == 1):
            i += 1
        elif (item == 0):
            n += 1
    perMal = (i/len(breastCancer)) * 100
    perBeg = (n/len(breastCancer)) * 100
    print("The percentage of Malignant Dx is: {0:.2f}%"\
      .format(perMal)) 
    print("The percentage of Begnin Dx is: {0:.2f}%"\
      .format(perBeg))
```

Let's check if this worked. I'm sure there's more effective ways of doing this process, but this is me doing brute-force attempts to defining/creating working functions. Don't worry I'll get better with practice :)

```python
classImbalance('diagnosis')
```
### Terminal Output 

    The percentage of Malignant Dx is: 37.26%
    The percentage of Begnin Dx is: 62.74%


As we can see here our data set is not suffering from *class imbalance* so we can proceed with our analysis. 


So I started by using the `.describe()` function to give some basic statistics relating to each variable. We can see there are 569 instances of each variable (which should make sense), but important to note that the distributions of the different variables have very high variance by looking at the **means** (Some can go as low as .0n while some as large as 800!)



```python
breastCancer.describe()
```
<!-- Missing Output -->

We will discuss the high variance in the distribution of the variables later within context of the appropriate analysis. For now we move on to visual representations of our data. Still a continuation of our **Exploratory Analysis**.

# Visual Exploratory Analysis <a name='visexploranal'></a>
For this section we utilize the module `Seaborn` which contains many powerful statistical graphs that would have been hard to produce using `matplotlib` (My note: `matplotlib` is not the most user friendly like `ggplot2` in *R*, but *Python* seems like its well on its way to creating visually pleasing and inuitive plots!)

## Scatterplot Matrix
For this visual I cheated by referencing some variables that were indicators of being influencial to the analysis (See **Random Forest** Section).


```python
# Scatterplot Matrix
# Variables chosen from Random Forest modeling.
cols = ['concave_points_worst', 'concavity_mean', 
        'perimeter_worst', 'radius_worst', 
        'area_worst', 'diagnosis']

sns.pairplot(breastCancer,
             x_vars = cols,
             y_vars = cols,
             hue = 'diagnosis', 
             palette = ('Red', '#875FDB'), 
             markers=["o", "D"])
```

<img src='https://raw.githubusercontent.com/raviolli77/machineLearning_breastCancer_Python/master/images/breastCancerWisconsinDataSet_MachineLearning_19_0.png'>

You see a matrix of the visual representation of the relationship between 6 variables:
+ `concave_points_worst`
+ `concavity_mean`
+ `perimeter_worst`
+ `radius_worst`
+ `area_worst`
+ `diagnosis`

Within each scatterplot we were able to color the two classes of **Dx**, which we can clearly see that we can easily distinguish the difference between **Malignant** and **Begnin**. As well as some variable interactions have an almost linear relationship. 

Of course these are just 2-dimensional representations, but its still interesting to see how variables interact with each other in our data set.  

## Pearson Correlation Matrix
The next visual that gives similar context to the last visual, this is called the *Pearson Correlation Matrix*. 

Variable correlation within a *Machine Learning* context doesn't play as an important role as say *linear regression*, there can still be some dangers when a data set has too many variables (and correlation between them). 


When two features (or more) are almost perfectly correlated in a *Machine Learning* setting then one has the potential to not add addition information to your process, thus feature extraction would help reduce dimensions which helps with the [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). As well as reducing computational time if once we deduce that feature extraction is necessary from visually seeing too much correlation within our variables.     




```python
corr = breastCancer.corr(method = 'pearson') # Correlation Matrix

f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 275, as_cmap=True)


# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,  cmap=cmap,square=True, 
            xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
```

<img src='https://raw.githubusercontent.com/raviolli77/machineLearning_breastCancer_Python/master/images/breastCancerWisconsinDataSet_MachineLearning_22_1.png'>

We can see that our data set contains mostly positive correlation, as well as re-iterating to us that the 5 dependent variables we featured in the *Scatterplot Matrix* have strong *correlation*. Our variables don't have too much correlation so I won't go about doing feature extraction processes like *Principal Component Analysis * (**PCA**), but you are more welcomed to do so (you will probably get better prediction estimates).    

## Boxplots
Next I decided to include boxplots of the data to show the high variance in the distribution of our variables. This will help drive home the point of the need to do some appropriate transformation for some models I will be employing. This is especially true for *Neural Networks*. 

```python
f, ax = plt.subplots(figsize=(11, 15))

ax.set_axis_bgcolor('#fafafa')
ax.set(xlim=(-.05, 50))
plt.ylabel('Dependent Variables')
plt.title("Box Plot of Pre-Processed Data Set")
ax = sns.boxplot(data = breastCancer, 
  orient = 'h', 
  palette = 'Set2')
```

<img src="https://raw.githubusercontent.com/raviolli77/machineLearning_breastCancer_Python/master/images/breastCancerWisconsinDataSet_MachineLearning_25_0.png">

Not the best picture but this is a good segue into the next step in our *Machine learning* process.

Here I used a function I created in my python script. Refer to `helperFunction.py` to understand the process but I'm setting the minimum of 0 and maximum of 1 to help with some machine learning applications later on in this report. 
```
# From helperFunction script 
def normalize_df(frame):
	'''
	Helper function to Normalize data set
	Intializes an empty data frame which 
	will normalize all floats types
	and just append the non-float types 
	so basically the class in our data frame
	'''
	breastCancerNorm = pd.DataFrame()
	for item in frame:
		if item in frame.select_dtypes(include=[np.float]):
			breastCancerNorm[item] = ((frame[item] - frame[item].min()) / 
			(frame[item].max() - frame[item].min()))
		else: 
			breastCancerNorm[item] = frame[item]
	return breastCancerNorm
```
Next we utilize the function on our dataframe. 

```python
breastCancerNorm = normalize_df(breastCancer)
```
Note that we won't use this dataframe until we start fitting *Neural Networks*. 


Let's try the `.describe()` function again and you'll see that all variables have a maximum of 1 which means we did our process correctly. 


```python
breastCancerNorm.describe()
```
<!-- Missing Output -->

### Box Plot of Transformed Data

Now to further illustrate the transformation let's create a *boxplot* of the scaled data set, and see the difference from our first *boxplot*. 


```python
f, ax = plt.subplots(figsize=(11, 15))

ax.set_axis_bgcolor('#fafafa')
plt.title("Box Plot of Transformed Data Set (Breast Cancer Wisconsin Data Set)")
ax.set(xlim=(-.05, 1.05))
ax = sns.boxplot(data = breastCancerNorm[1:29], 
  orient = 'h', 
  palette = 'Set2')
```


<img src="https://raw.githubusercontent.com/raviolli77/machineLearning_breastCancer_Python/master/images/breastCancerWisconsinDataSet_MachineLearning_34_0.png">

There are different forms of transformations that are available for *machine learning* and I suggest you research them to gain a better understanding as to when to use a transformation. 
But for this project I will only employ the transformed dataframe on *Neural Networks*. 

# Model Estimation <a name='modelest'></a>
Now that we've gotten a *feel* for the data, we are ready to begin predictive modeling. When going about predicitive modeling, especially when new data isn't readily available, creating a *training* and *test sets* will help in understanding your model's predictive power. 

We will use the *training set* to train our model, essentially learning from data it's seeing to later infer data it hasn't seen yet. We want to avoid using our entire data set on the training process, due to the process called *overfitting*. 

### Overfitting
We run the risk of *over-fitting* our data when we train the model on our entire dataset, especially true for this data set since we don't have any other data to see how well our data does. *Over-fitting* will cause our model to output strong predicitive power, but only for our training data.  

We avoid this through the use of the *training* and *test set*, where we measure the predicitive power on the *test set*. This will be a good indicator of our model's performance, but test sets also have their limitations. This is where *Cross Validation* will come into play, but more on this later. 

## Creating Training and Test Sets

We split the data set into our training and test sets which will be (pseudo) randomly selected having a *80-20%* splt. 

**NOTE**: What I mean when I say pseudo-random is that we would want everyone who replicates this project to get the same results especially if you're trying to learn from this project. So we use a random seed generator and set it equal to a number of your choosing, which will then make the results for anyone who uses this generator receive the same exact results. 


```python
# Here we do a 80-20 split for our training and test set
train, test = train_test_split(breastCancer, 
                               test_size = 0.20, 
                               random_state = 42)

# Create the training test omitting the diagnosis
training_set = train.ix[:, train.columns != 'diagnosis']
# Next we create the class set (Called target in Python Documentation)
# Note: This was confusing af to figure out 
# cus the documentation is low-key kind of shitty
class_set = train.ix[:, train.columns == 'diagnosis']

# Next we create the test set doing the same process as the training set
test_set = test.ix[:, test.columns != 'diagnosis']
test_class_set = test.ix[:, test.columns == 'diagnosis']
```

# Kth Nearest Neighbor <a name='knn'></a>

A popular algorithm within classification, *kth nearest neighbor* employs a majority votes methodology using *Euclidean Distance* (a.k.a straight-line distance between two points) based on the specificied *k*. 

So within context of my data set, I employ *k=7* so the algorithm looks for the 7 neighbors closest to the value its trying to classify (again closest measured using [Euclidean Distance](https://mathworld.wolfram.com/Distance.html)). This algorithm is known as a *lazy algorithm* and is considered one of the simpler algorithms in *Machine Learning*. 


One can often find that *K-nn* will perform better than more complicated methods, recall *Occam's Razor* when going about data analysis. 

**Important to Note**: The biggest drawback for *K-NN* is the [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)  

## Determining Optimal k 
Determining the best *k* can be done mathematically by creating a `for-loop` that searches through multiple *k* values, we collect the `cross_val` scores which we then see which was the lowest *mean squared error* and receive the corresponding *k*. 
This section was heavily influenced by [Kevin Zakka's Blog Post](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/) 

```
myKs = []
for i in range(0, 50):
    if (i % 2 != 0):
        myKs.append(i)

cross_vals = []
for k in myKs:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,
                             training_set, 
                             class_set['diagnosis'], 
                             cv = 10, 
                             scoring='accuracy')
    cross_vals.append(scores.mean())

MSE = [1 - x for x in cross_vals]
optimal_k = myKs[MSE.index(min(MSE))]
print("Optimal K is {0}".format(optimal_k))
```
### Terminal Output

	Optimal K is 7

### Set up the Algorithm
I will be employing `scikit-learn` algorithms for this process of my project, but in the future I would be interested in building the algorithms from "scratch". I'm not there yet with my learning though. 


```python
fit_KNN = KNeighborsClassifier(n_neighbors=7)
```

### Train the Model

Next we train the model on our *training set* and the *class set*. The other models follow a similar pattern, so its worth noting here I had to call up the `'diagnosis'` column in my `class_set` otherwise I got an error code. 


```python
fit_KNN.fit(training_set, class_set['diagnosis'])
```

### Terminal Output 


    KNeighborsClassifier(algorithm='auto', leaf_size=30, 
               metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=7, p=2,
               weights='uniform')



### Training Set Calculations
For this set, we calculate the accuracy based on the training set. I decided to include this to give context, but within *Machine Learning* processes we aren't concerned with the *training error rate* since it can suffer from high *bias* and over-fitting. 

As I stated previously we're more concerned in seeing how our model does against data it hasn't seen before. 


```python
# We predict the class for our training set
predictionsTrain = fit_KNN.predict(training_set) 

# Here we create a matrix comparing the actual values vs. the predicted values
print(pd.crosstab(predictionsTrain, 
                  class_set['diagnosis'], 
                  rownames=['Predicted Values'], 
                  colnames=['Actual Values']))

# Measure the accuracy based on the trianing set
accuracyTrain = fit_KNN.score(training_set, class_set['diagnosis'])

print("Here is our accuracy for our training set: {0: .3f} "\
  .format(accuracyTrain))
```
### Terminal Output 

    Actual Values       0    1
    Predicted Values          
    0                 279   20
    1                   7  149
    Here is our accuracy for our training set:  0.941 


Here we get the training set error from the `.score()` function where test set calculations will follow a similar workflow. 
```python
train_error_rate = 1 - accuracyTrain   
print("The train error rate for our model is: {0: .3f}"\
.format(train_error_rate))
```
### Terminal Output 

    The train error rate for our model is:  0.059


## Cross Validation 
Cross validation is a powerful tool that is used for estimating the predicitive power of your model, which performs better than the conventional *training* and *test set*. What we are doing with *Cross Validation* is we are essentially creating multiple *training* and *test sets*. 

In our case we are creating 10 sets within our data set that calculates the estimations we have done already, but then averages the prediction error to give us a more accurate representation of our model's prediction power, since the model's performance can vary significantly when utilizing different *training* and *test sets*. 

**Suggested Reading**: For a more concise explanation of *Cross Validation* I recommend reading *An Introduction to Statistical Learnings with Applications in R*, specifically chapter 5.1!


## K-Fold Cross Validation 
Here we are employing *K-Fold Cross Validation*, more specifically 10 folds. So therefore we are creating 10 subsets of our data where we will be employing the *training* and *test set methodology* then averaging the accuracy for all folds to give us our estimatation. 

We could have done *Cross Validation* on our entire data set, but when it comes to tuning our parameters (more on this later) we still need to utilize a *test set*. This is more important for other models, since the only part we really have to optimize for *Kth Nearest Neighbors* is the *k*, but for the sake of consistency we will do *Cross Validation* on the *training set*.
```python
n = KFold(n_splits=10)

scores = cross_val_score(fit_KNN, 
                         test_set, 
                         test_class_set['diagnosis'], 
                         cv = n)
print("Accuracy: {0: 0.2f} (+/- {1: 0.2f})"\
      .format(scores.mean(), scores.std() / 2))
```

### Terminal Output 

    0.966 (+/- 0.021)


## Test Set Evaluations

Next we start doing calculations on our *test set*, this will be our form of measurement that will indicate whether our model truly will perform as well as it did for our *training set*. 

We could have simply over-fitted our model, but since the model hasn't seen the *test set* it will be an un-biased measurement of accuracy. 


```python
# First we predict the Dx for the test set and call it predictions
predictions = fit_KNN.predict(test_set)

# Let's compare the predictions vs. the actual values
print(pd.crosstab(predictions, 
                  test_class_set['diagnosis'], 
                  rownames=['Predicted Values'], 
                  colnames=['Actual Values']))

# Let's get the accuracy of our test set
accuracy = fit_KNN.score(test_set, test_class_set['diagnosis'])

# TEST ERROR RATE!!
print("Here is our accuracy for our test set: {0: .3f}"\
  .format(accuracy))
```

### Terminal Output 

    Actual Values      0   1
    Predicted Values        
    0                 70   4
    1                  1  39
    Here is our accuracy for our test set:  0.956



```python
# Here we calculate the test error rate!
test_error_rate = 1 - accuracy
print("The test error rate for our model is: {0: .3f}"\
  .format(test_error_rate))
```
### Terminal Output 

    The test error rate for our model is:  0.044


### Conclusion for K-NN

So as you can see our **K-NN** model did pretty well! The biggest set back was the number of samples it predicted to not have cancer when they actually had cancer. This will be a useful measurement as well since we are concerned with our models incorrectly predicting *begnin* when in actuality the sample is *malignant*. Since this would spell life-threatening mistakes if our models were to be applied in a real life situation. 

So we will keep this in mind when comparing models.

### Calculating for later use in ROC Curves

Here we calculate the *false positive rate* and *true positive rate* for our model. This will be used later when creating **ROC Curves** when comparing our models. I will go into more detail in a later section. 


```python
fpr, tpr, _ = roc_curve(predictions, test_class_set)
```

###  Calculations for Area under the Curve 
This function calculates the area under the curve which you will see the relevancy later in this project, but ideally we want it to be closest to 1 as possible. 


```python
auc_knn = auc(fpr, tpr)
```

# Random Forest <a name='randforest'></a>
Also known as *Random Decision Forest*, *Random Forest* is an entire forest of random uncorrelated decision trees. This is an extension of *Decision Trees* that will perform significantly better than a single tree because it corrects over-fitting. Here is a brief overview of the evolution of *CART* analysis:
+ Single Decision Tree (Single tree)
+ Bagging Trees (Multiple trees) [Model with all features, M, considered at splits, where M = all features]
+ Random Forest (Multiple trees) [Model with m features considered at splits, where m < M, typically m = sqrt(M)]

### Bagging Trees
*Decision Trees* tend to have *low bias and high variance*, a process known as *Bagging Trees* (*Bagging* = *Bootstrap Aggregating*) was an extension that does random sampling with replacement where after creating N trees it classifies on majority votes. This process reduces the variance while at the same time keeping the bias low. However, a downside to this process is if certain features are strong predictors then too many trees will employ these features causing correlation between the trees. 

Thus *Random Forest* aims to reduce this correlation by choosing only a subsample of the feature space at each split. Essentially aiming to make the trees more independent thereby reducing the variance.   

Generally, we aim to create 500 trees and use our m to be sqrt(M) rounded down. So since we have 30 features I will use 5 for my `max_features` parameter. I will be using the *Entropy Importance* metric. 

For this model we use an index known as the [Information Gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees). 


### Variable Importance

A useful feature within what are known as *CART* (Classification And Regression Trees) is extracting which features where the most important when using *Entropy* within context of *information gain*. 

For this next process we will be grabbing the index of these features, then using a `for loop` to state which were the most important. 


```python
fit_RF = RandomForestClassifier(random_state = 42, 
                                criterion='gini',
                                n_estimators = 500,
                                max_features = 5)
```


```python
fit_RF.fit(training_set, class_set['diagnosis'])
```

### Terminal Output 


    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features=5, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=500, n_jobs=1, oob_score=False, random_state=42,
                verbose=0, warm_start=False)



## Variable Importance
We can gain information from our model through the concept of *variable importance*. This allows us to see which variables played an important role in the forest that was created. 

Using the `.feature_importances_` feature in **CART** models, we extract the index of our feature space then do a `for-loop` to receive the variables that had the most importance when using the *entropy* criterion. 
```python
importancesRF = fit_RF.feature_importances_
indicesRF = np.argsort(importancesRF)[::-1]
indicesRF
```

### Terminal Output 


    array([27, 23,  7, 22, 20,  6,  0,  2,  3, 26, 13, 21, 10,  1, 25, 28, 24,
            5, 12, 16,  4, 19, 29, 15, 18,  9, 17, 11,  8, 14])



This will give us the importance of the variables from largest to smallest, which we will then visualize. 
```python
# Print the feature ranking
print("Feature ranking:")

for f in range(30):
    i = f
    print("%d. The feature '%s' \
	has a Information Gain of %f" % (f + 1,
				namesInd[indicesRF[i]],
				importancesRF[indicesRF[f]]))
```

### Terminal Output 

    Feature ranking:
    1. The feature 'concave_points_worst' has an Information Gain of 0.139713
    2. The feature 'area_worst' has an Information Gain of 0.122448
    3. The feature 'concave_points_mean' has an Information Gain of 0.115332
    4. The feature 'perimeter_worst' has an Information Gain of 0.114410
    5. The feature 'radius_worst' has an Information Gain of 0.082506
    6. The feature 'concavity_mean' has an Information Gain of 0.051091
    7. The feature 'radius_mean' has an Information Gain of 0.047065
    8. The feature 'perimeter_mean' has an Information Gain of 0.041769
    9. The feature 'area_mean' has an Information Gain of 0.040207
    10. The feature 'concavity_worst' has an Information Gain of 0.038435
    11. The feature 'area_se' has an Information Gain of 0.029797
    12. The feature 'texture_worst' has an Information Gain of 0.021006
    13. The feature 'radius_se' has an Information Gain of 0.016963
    14. The feature 'texture_mean' has an Information Gain of 0.016359
    15. The feature 'compactness_worst' has an Information Gain of 0.015939
    16. The feature 'symmetry_worst' has an Information Gain of 0.013319
    17. The feature 'smoothness_worst' has an Information Gain of 0.013109
    18. The feature 'compactness_mean' has an Information Gain of 0.012102
    19. The feature 'perimeter_se' has an Information Gain of 0.010395
    20. The feature 'concavity_se' has an Information Gain of 0.007546
    21. The feature 'smoothness_mean' has an Information Gain of 0.007518
    22. The feature 'fractal_dimension_se' has an Information Gain of 0.006149
    23. The feature 'fractal_dimension_worst' has an Information Gain of 0.005899
    24. The feature 'compactness_se' has an Information Gain of 0.005324
    25. The feature 'symmetry_se' has an Information Gain of 0.004980
    26. The feature 'fractal_dimension_mean' has an Information Gain of 0.004723
    27. The feature 'concave_points_se' has an Information Gain of 0.004516
    28. The feature 'texture_se' has an Information Gain of 0.004405
    29. The feature 'symmetry_mean' has an Information Gain of 0.003570
    30. The feature 'smoothness_se' has an Information Gain of 0.003402


The same process can be done for any **CART** when looking at the *Variable Importance*.

### Feature Importance Visual

Here I use the `sorted` function to sort the *Information Gain* criterion from least to greatest which was a work around in order to create a horizontal barplot, as well as creating an index using the `arange` function in `numpy`


```python
indRf = sorted(importancesRF) # Sort by Decreasing order
index = np.arange(30)
index
```
### Terminal Output 



    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])



This next step we're creating an index that's backwards since we're creating a barplot that is horizontal we have to state the correct order. If we don't do this step then the names will be in reverse order (we don't want that).
```python
feature_space = []
for i in range(29, -1, -1):
    feature_space.append(namesInd[indicesRF[i]])
```

Now let's plot it. 


```python
f, ax = plt.subplots(figsize=(11, 11))

ax.set_axis_bgcolor('#fafafa')
plt.title('Feature importances for Random Forest Model')
plt.barh(index, indRf,
        align="center", 
        color = '#875FDB')

plt.yticks(index, feature_space)
plt.ylim(-1, 30)
plt.xlim(0, 0.15)
plt.xlabel('Gini Importance')
plt.ylabel('Feature')

plt.show()
```
<img src='https://raw.githubusercontent.com/raviolli77/machineLearning_breastCancer_Python/master/images/varImprt.png'>

## Cross Validation 
Again here we employ a 10 fold *cross validation* method to get another accuracy estimation for our models. 

```python
n = KFold(n_splits=10)
scores = cross_val_score(fit_RF, 
                         test_set, 
                         test_class_set['diagnosis'], 
                         cv = n)

print("Accuracy: {0: 0.2f} (+/- {1: 0.2f})"\
      .format(scores.mean(), scores.std() / 2))
```
### Terminal Output 

    Accuracy:  0.96 (+/-  0.03)


## Test Set Evaluations
These processes are similar to the previous *test set evaluation* so explanations won't be necessary. 

```python
predictions_RF = fit_RF.predict(test_set)
```


```python
print(pd.crosstab(predictions_RF, 
                  test_class_set['diagnosis'], 
                  rownames=['Predicted Values'], 
                  colnames=['Actual Values']))
```
### Terminal Output 

    Actual Values      0   1
    Predicted Values        
    0                 70   3
    1                  1  40



```python
accuracy_RF = fit_RF.score(test_set, test_class_set['diagnosis'])

print("Here is our mean accuracy on the test set:\n {0:.3f}"\
      .format(accuracy_RF))
```
### Terminal Output 

    Here is our mean accuracy on the test set:
     0.965



```python
# Here we calculate the test error rate!
test_error_rate_RF = 1 - accuracy_RF
print("The test error rate for our model is:\n {0: .3f}"\
      .format(test_error_rate_RF))
```
### Terminal Output 

    The test error rate for our model is:
      0.035


### Calculating for later use in ROC Curves


```python
fpr2, tpr2, _ = roc_curve(predictions_RF, 
                          test_class_set)
```

### Calculations for  Area under Curve


```python
auc_rf = auc(fpr2, tpr2)
```

## Conclusions for Random Forest
Our *Random Forest* performed pretty well, I have a personal preference to tree type models because they are the most interprettable and give insight to data that some other models don't (for instance **K-NN**). We were able to see which variables were important when the *random forest* was created and thus we can expand on our data/model if we choose too, through the variable importance of *random forest*. 
That can be an exercise for later on to see if choosing a subset of the data set will help in prediction power. For now I am content with using the entire data set for instructive purposes. 

Important to note, is that the *random forest* performed better in terms of *false negatives*, only slightly though. 
# Neural Network <a name='neurnetwork'></a>
Our last model is a harder concept to understand. *Neural Networks* are considered *black box models*, meaning that we don't know what goes on under the hood, we feed the model our data and it outputs the predictions without giving us insight into how our model works. 

*Neural Network* architecture is very complex and I will do no justice explaining it in a few sentences, but I will try to give a brief overview to help with giving context. 
So there are three main components that make up an *artificial neural network*:
+ **Input Layer**: This is where we feed data into the model, we have a *neuron* for every variable in our feature space (in our case we have 30 input layers)
+ **Hidden Layer(s)**: This is where the magic happens, we have several hidden layers with activation functions (typically logisitic activation functions) that create weights and corrects itself through [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) and [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) for further optimization
	+ **Backpropagation**: The inputs are fed into the *neural network*, so they go through the hidden layers until they reach the output layer (forward propogation), which then outputs a prediction, the model then compares the prediction to the actual values utilizing a [loss function](https://en.wikipedia.org/wiki/Loss_function) to go backwards to optimize the hidden layers (through weights) using *gradient descent*
	+ **Gradient Descent**: 
+ **Output Layer**: This section represents the output of the model so in our data set our output layer is binary

Here you can see a visual representation of a simple *neural network*:

<img src='http://cs231n.github.io/assets/nn1/neural_net.jpeg'>

We use the normalized dataframe that we created earlier to fit the *Neural Network*. Important to note is that *neural networks* require a lot of pre-processing in order for them to be effective models. The standard is to standardize your dataframe, or else the model will perform noticeably poor. 

So here we are creating a new *training* and *test set* to input into our *neural network*. 

```python
training_set_norm, class_set_norm, test_set_norm, test_class_set_norm = splitSets(breastCancerNorm)
```
Here we create our *neural network* utilizing 5 hidden layers, the logisitic activiation function, and 
We won't be using *Gradient Descent* in this model instead we will be utilizing `lbfgs` which is an optimizer from quasi-Newton methods family, which according to the documentation:

*"For small datasets, however, ‘lbfgs’ can converge faster and perform better."* [Source](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

Since our data set is relatively small I decided to go with `lbfgs`.

```python
fit_NN = MLPClassifier(solver='lbfgs', 
                       hidden_layer_sizes=(5, ), 
                       activation='logistic',
                       random_state=7)
```
## Train the Model 
Similar to the past two other models we train our model utilizing the *training set*. 

```python
fit_NN.fit(training_set_norm, 
           class_set_norm['diagnosis'])
```
### Terminal Output 



    MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(5,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=7, 
           shuffle=True, solver='lbfgs', tol=0.0001, 
           validation_fraction=0.1, verbose=False,
           warm_start=False)


## Cross Validation 
Likewise we are utilizing 10 fold *cross validation* for this model. 

```python
n = KFold(n_splits=10)
scores = cross_val_score(fit_NN, 
                         test_set_norm, 
                         test_class_set_norm['diagnosis'], 
                         cv = n)

print("Accuracy: {0: 0.2f} (+/- {1: 0.2f})"\
      .format(scores.mean(), scores.std() / 2))
```
### Terminal Output 

    Accuracy:  0.93 (+/-  0.03)


Our model performed significantly worse then the other two models. Let's calculate the *test error rate* to compare its performance. 
## Test Set Calculations
Let's do the same test set calculations and compare all our models. 
```
predictions_NN = fit_NN.predict(test_set_norm)

print(pd.crosstab(predictions_NN, test_class_set_norm['diagnosis'], 
                  rownames=['Predicted Values'], 
                  colnames=['Actual Values']))```


### Terminal Output

```
Actual Values      0   1
Predicted Values        
0                 68   1
1                  3  42
```
The *Neural Network* model performed better in terms of *false negative*!
 
```python
accuracy_NN = fit_NN.score(test_set_norm, test_class_set_norm['diagnosis'])

print("Here is our mean accuracy on the test set:\n{0: .2f} "\
  .format(accuracy_NN))
```
### Terminal Output 

    Here is our mean accuracy on the test set:
     0.965



```python
# Here we calculate the test error rate!
test_error_rate_NN = 1 - accuracy_NN
print("The test error rate for our model is:\n {0: .3f}"\
      .format(test_error_rate_NN))
```
### Terminal Output 

    The test error rate for our model is:
       0.035 



```python
fpr3, tpr3, _ = roc_curve(predictions_NN, test_class_set)
```


```python
auc_nn = auc(fpr3, tpr3)
```
## Conclusions for Neural Networks 
In terms of *cross validation*, our *neural network* peformed worse then the other two models. But this was expected since *neural networks* require a lot of *training data* to perform at its best, and unfortunately for us we only have the data that was provided in the *UCI repository*. 

# ROC Curves <a name='roccurve'></a>

*Receiver Operating Characteristc* Curve calculations we did using the function `roc_curve` were calculating the **False Positive Rates** and **True Positive Rates** for each model. We will now graph these calculations, and being located the top left corner of the plot indicates a really ideal model, i.e. a **False Positive Rate** of 0 and **True Positive Rate** of 1, so we plot the *ROC Curves* for all our models in the same axis and see how they compare!



We also calculated the **Area under the Curve**, so our curve in this case are the *ROC Curves*, we then place these calculations in the legend with their respective models. 


```python
f, ax = plt.subplots(figsize=(10, 10))

plt.plot(fpr, tpr, label='K-NN ROC Curve (area = {0: .3f})'.format(auc_knn), 
         color = 'deeppink', 
         linewidth=1)

plt.plot(fpr2, tpr2,label='Random Forest ROC Curve (area = {0: .3f})'\
	.format(auc_rf), 
         color = 'red', 
         linestyle=':', 
         linewidth=2)

plt.plot(fpr3, tpr3,label='Neural Networks ROC Curve (area = {0: .3f})'\
	.format(auc_nn), 
         color = 'purple', 
         linestyle=':', 
         linewidth=3)
	
ax.set_axis_bgcolor('#fafafa')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison For All Models')
plt.legend(loc="lower right")
	
plt.show()
```


<img src="https://raw.githubusercontent.com/raviolli77/machineLearning_breastCancer_Python/master/images/rocCurve.png">

Let's zoom in to get a better picture!

### ROC Curve Plot Zoomed in


```python
f, ax = plt.subplots(figsize=(10, 10))
plt.plot(fpr, tpr, label='K-NN ROC Curve  (area = {0: .3f})'\
         .format(auc_knn), 
         color = 'deeppink', 
         linewidth=1)

plt.plot(fpr2, tpr2,label='Random Forest ROC Curve  (area = {0: .3f})'\
	.format(auc_rf), 
         color = 'red', 
         linestyle=':', 
         linewidth=3)

plt.plot(fpr3, tpr3,label='Neural Networks ROC Curve  (area = {0: .3f})'\
	.format(auc_nn), 
         color = 'purple', 
         linestyle=':', 
         linewidth=3)
	
ax.set_axis_bgcolor('#fafafa')
plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
	
plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
plt.xlim([-0.001, 0.2])
plt.ylim([0.7, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison For All Models (Zoomed)')
plt.legend(loc="lower right")
	
plt.show()
```


<img src="https://raw.githubusercontent.com/raviolli77/machineLearning_breastCancer_Python/master/images/rocZoom.png">

Visually examining the plot, *Random Forest* and *Kth Nearest Neighbor* are noticeable more elevated than *Neural Networks* which is indicative of a good prediction tool, using this form of diagnositcs. 

# Conclusions <a name='conclusions'></a>
Once I employed all these methods, we can that **Random Forest** performed the best in terms of most diagnostics, as well as giving us insight into the feature space that the other two models can't provide. 

When choosing models it isn't just about having the best accuracy, we are also concerned with insight because these models can help tell us which dependent variables are indicators thus helping researchers in the respective field to focus on the variables that have the most statistically significant influence in predictive modeling within the respective domain. *Kth Nearest Neighbor* performed better in terms of *cross validation*, but I have yet to perform *hyperparameter optimization* on other processes.  

This project is an iterative process, so I will be working to reach a final consensus. In terms of most insight into the data, *random forest* model is able to tell us the most of our model. In terms of cross validated performance *kth nearest neighbor* performed the best.  

### Diagnostics for Data Set

| Model/Algorithm 	| Test Error Rate 	| False Negative for Test Set 	| Area under the Curve for ROC | Cross Validation Score | Hyperparameter Optimization | 
|-----------------|-----------------|-------------------------------|----------------------------|-----------|------|
| Kth Nearest Neighbor | 0.035 |	2 |	0.963 | 0.966 (+/-  0.021) | Optimal **k** = 7 |
| Random Forest 	|  0.035	| 3 	| 0.9673 |  0.955 (+/-  0.022) |  {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 4} | 
| Neural Networks 	| 0.035 	| 1 	| 0.952 |  0.947 (+/-  0.030) |  {'hidden_layer_sizes': 12, 'activation': 'tanh', 'learning_rate_init': 0.05} | 


# Sources Cited

+ W. Street, Nick. UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
+ Waidyawansa, Buddhini. Kaggle [https://www.kaggle.com/]. *Using the Wisconsin breast cancer diagnostic data set for predictive analysis*. [Kernel Source](https://www.kaggle.com/buddhiniw/breast-cancer-prediction)
+ Zakka, Kevin. Kevin Zakka's Blog [https://kevinzakka.github.io/]. *A Complete Guide to K-Nearest-Neighbors with Applications in Python and R*. [Blog Source](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/)