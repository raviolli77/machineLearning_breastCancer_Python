
## Breast Cancer Machine Learning Techniques
Machine Learning with Breast Cancer data set. 

+ Raul Eulogio

## Abstract
For this quick project, I wanted to implement a few **Machine Learning** techniques on a data set containing descriptive attributes of digitized images of a process known as, fine needle aspirate (**FNA**) of breast mass. We have a total of 29 features that were computed for each cell nucleus with an ID Number and the Diagnosis (Later converted to binary representations: **Malignant** = 1, **Benign** = 0). 

I used the same models as the other notebook, but this is the expanded data set, and goes more in-depth with the explanations for this project!

**UPDATES**

<!--
(2/3/2017):
+ Exploratory analysis
+ Better ROC Curve Visuals 
+ More Comments

(2/17/2017):
+ Comments on models (Kth Nearest Neighbor)
+ More succint data analysis processes

(2/18/2017):
+ Comments on models (Decision Trees, Random Forest)
+ Variable Importance Visual for Random Forest
+ Better ROC Curve Visuals (Added dotted axis as well as auc Calculations) 

(3/17/2017):
+ Changed Neural Network settings to match .py script
+ Changed ROC Curves accordingly
-->
<img src="https://www.researchgate.net/profile/Syed_Ali39/publication/41810238/figure/fig5/AS:281736006127621@1444182506838/Figure-2-Fine-needle-aspiration-of-a-malignant-solitary-fibrous-tumor-is-shown-A-A.png">


Ex. Image of a malignant solitary fibrous tumor using **FNA**

This is popular data set used for machine learning purposes, and I plan on using the same techniques I used for another data set that performed poorly due to having too many *Categorical* variables (**NOTE**: Learned about *dummy variables* so might revisit and execute analysis correctly this time!)

Here are the **Machine learning methods** I decided to use:

+ Random Forest
+ Neural Networks

I employ critical data analysis modules in this project, emphasizing on: 

+ pandas
+ scikit learn 
+ matplotlib (for visuals)
+ seaborn (easier to make statistical plots)

## 1. Load Modules
We load our modules into our python environment. In my case I am employing a **Jupyter Notebook** while running inside an **|conda** environment. 

For now to illustrate and show the module versions in a simple way I will name the ones I used and show the version I used as follows:

+ numpy==1.11.2
+ pandas==0.18.1
+ matplotlib==1.5.3
+ sklearn==0.18.1
+ seaborn=0.7.1


```python
%matplotlib inline

import numpy as np
import pandas as pd # Data frames
import matplotlib.pyplot as plt # Visuals
import seaborn as sns # Danker visuals
from helperFunctions import *
from sklearn.model_selection import train_test_split # Create training and test sets
from sklearn.model_selection import KFold, cross_val_score # Cross validation
from sklearn.neighbors import KNeighborsClassifier # Kth Nearest Neighbor
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.tree import export_graphviz # Extract Decision Tree visual
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.neural_network import MLPClassifier # Neural Networks
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # Calculating Area Under Curve for ROC's!
from urllib.request import urlopen # Get data from UCI Machine Learning Repository

pd.set_option('display.max_columns', 500) # Included to show all the columns 
# since it is a fairly large data set
plt.style.use('ggplot') # Using ggplot2 style visuals because that's how I learned my visuals 
# and I'm sticking to it!
```

## Loading Data
For this section, I'll load the data into a **Pandas** dataframe using `urlopen` from the `urllib.request` module. 

Instead of downloading a csv, I started implementing this method(Inspired by Jason's Python Tutorials) where I grab the data straight from the **UCI Machine Learning Database**. Makes it easier to go about analysis from online sources and cuts out the need to download/upload a csv file when uploading on **GitHub**. I create a list with the appropriate names and set them within the dataframe. **NOTE**: The names were not documented to well so I used [this analysis](https://www.kaggle.com/buddhiniw/d/uciml/breast-cancer-wisconsin-data/breast-cancer-prediction) to grab the variable names and some other tricks that I didn't know that were available in **Python** (I will mention the use in the script!)

Finally I set the column `id_number` as the index for the dataframe. 


```python
UCI_data_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

names = ['id_number', 'diagnosis', 'radius_mean', 
         'texture_mean', 'perimeter_mean', 'area_mean', 
         'smoothness_mean', 'compactness_mean', 'concavity_mean',
         'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
         'radius_se', 'texture_se', 'perimeter_se', 
         'area_se', 'smoothness_se', 'compactness_se', 
         'concavity_se', 'concave_points_se', 
         'symmetry_se', 'fractal_dimension_se', 
         'radius_worst', 'texture_worst', 'perimeter_worst',
         'area_worst', 'smoothness_worst', 
         'compactness_worst', 'concavity_worst', 
         'concave_points_worst', 'symmetry_worst', 
         'fractal_dimension_worst'] 

breastCancer = pd.read_csv(urlopen(UCI_data_URL), names=names)

breastCancer.set_index(['id_number'], inplace = True) # Setting 'id_number' as our index
namesInd = names[2:] # FOR CART MODELS LATER
```

### Outputting our data
Its good to always output sections of your data so you can give context to the reader as to what each column looks like, as well as seeing examples of how the data is suppose to be formatted when loaded correctly. Many people run into the issue (especially if you run into a data set with poor documentation w.r.t. the column names), so its good habit to show your data during your analysis. 

We use the function `head()` which is essentially the same as the `head` function in **R** if you come from an **R** background. Notice the syntax for **Python** is significantly different than that of **R** though. 


```python
breastCancer.head()
```
<!-- Missing output -->

### More Preliminary Analysis
Much of these sections are given to give someone context to the dataset you are utilizing. Often looking at raw data will not give people the desired context, so it is important for us as data enthusiast to fill in the gaps for people who are interested in the analysis. But don't plan on running it anytime soon. 

#### Data Frame Dimensions
Here we use the `.shape` function to give us the lengths of our data frame, where the first output is the row-length and the second output is the column-length. 

#### Data Types
Another important piece of information that is **important** is the data types of your variables in the data set. 

It is often good practice to check the variable types with either the source or with your own knowledge of the data set. For this data set, we know that all variables are measurements, so they are all continous (Except **Dx**), so no further processing is needed for this step.

An common error that can happen is: if a variable is *discrete* (or *categorical*), but has a numerical representation someone can easily forget the pre-processing and do analysis on the data type as is. Since they are numeric they will be interpretted as either `int` or `float` which is incorrect. I can go on and on, but for this data set the numerical representation of the **Dx** is correct and is referred to as *indicator* or *dummy* variables. 




```python
print("Here's the dimensions of our data frame:\n", 
     breastCancer.shape)
print("Here's the data types of our columns:\n",
     breastCancer.dtypes)
```

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
Important when doing analysis, is converting variable types to the appropriate representation. A tool is as useful as the person utilizing it, so if we enter our data incorrectly the algorithm will suffer not as a result from its capabilities, but from the human component (More on this later). 

Here I converted the Dx to **binary** represenations using the `map` functionality in `pandas`. We are using a dictionary:

    {'M':1, 'B':0}

which then converts the previous string representations of the Dx to the **binary** representation, where 1 == **Malignant** and 0 == **Benign**. 


```python
# Converted to binary to help later on with models and plots
breastCancer['diagnosis'] = breastCancer['diagnosis'].map({'M':1, 'B':0})

# Let's look at the count of the new representations of our Dx's
breastCancer['diagnosis'].value_counts()
```




    0    357
    1    212
    Name: diagnosis, dtype: int64



## Class Imbalance
The count for our Dx is important because it brings up the discussion of *Class Imbalance* within *Machine learning* and *data mining* applications. 

*Class Imbalance* refers to when a class within a data set is outnumbered by the other class (or classes). 
Reading documentation online, *Class Imbalance* is present when a class populates 10-20% of the data set. 

However for this data set, its pretty obvious that we don't suffer from this, but since I'm practicing my **Python**, I decided to experiment with `functions` to get better at **Python**! 


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
    print("The percentage of Malignant Dx is: {0:.2f}%".format(perMal)) 
    print("The percentage of Begnin Dx is: {0:.2f}%".format(perBeg))
```

Let's check if this worked. I'm sure there's more effective ways of doing this process, but this is me doing brute-force attempts to defining/creating working functions. Don't worry I'll get better with practice :)

**OUTPUT**:


```python
classImbalance('diagnosis')
```

    The percentage of Malignant Dx is: 37.26%
    The percentage of Begnin Dx is: 62.74%


As we can see here our data set is not suffering from *class imbalance* so we can proceed with our analysis. 

# Exploratory Analysis

An important process in **Machine Learning** is doing **Exploratory Analysis** to get a *feel* for your data. As well as creating visuals that can be digestable for anyone of any skill level. 

So I started by using the `.describe()` function to give some basic statistics relating to each variable. We can see there are 569 instances of each variable (which should make sense), but important to note that the distributions of the different variables have very high variance by looking at the **means** (Some can go as low as .0n while some as large as 800!)



```python
breastCancer.describe()
```
<!-- Missing Output -->

We will discuss the high variance in the distribution of the variables later within context of the appropriate analysis. For now we move on to visual representations of our data. Still a continuation of our **Exploratory Analysis**.

# Visual Exploratory Analysis
For this section we utilize the module `Seaborn` which contains many powerful statistical graphs that would have been hard to produce using `matplotlib` (My note: `matplotlib` is not the most user friendly like `ggplot2` in **R**, but **Python** seems like its well on its way to creating visually pleasing and inuitive plots!)

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




    <seaborn.axisgrid.PairGrid at 0x7fb6ee6bfe48>




<img src=''>

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


When two features (or more) are almost perfectly correlated in a *Machine Learning* setting then one doesn't add any information to your process, thus feature extraction would help reduce dimensions (and/or remove variables that don't add to our model) which helps avoid the [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). As well as reducing computational time if once we deduce that feature extraction is necessary from visually seeing too much correlation within our variables.     




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




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb6f6418a20>




![png](breastCancerMachineLearningPresentation_files/breastCancerMachineLearningPresentation_19_1.png)


We can see that our data set contains mostly positive correlation, as well as re-iterating to us that the 5 dependent variables we featured in the *Scatterplot Matrix* have strong *correlation*. Our variables don't have too much correlation so I won't go about doing feature extraction processes like **PCA**, but you are more welcomed to do so (you will probably get better prediction estimates).    

We can see that our data set contains mostly positive correlation, as well as re-iterating to us that the 5 dependent variables we featured in the *Scatterplot Matrix* have strong *correlation*. Our variables don't have too much correlation so I won't go about doing feature extraction processes like **PCA**, but you are more welcomed to do so (you will probably get better prediction estimates).    


```python
f, ax = plt.subplots(figsize=(11, 15))

ax.set_axis_bgcolor('#fafafa')
ax.set(xlim=(-.05, 50))
plt.ylabel('Dependent Variables')
plt.title("Box Plot of Pre-Processed Data Set")
ax = sns.boxplot(data = breastCancer, orient = 'h', palette = 'Set2')
```


![png](breastCancerMachineLearningPresentation_files/breastCancerMachineLearningPresentation_21_0.png)


Not the best picture but this is a good segue into the next step in our *Machine learning* process.

Here I used a function I created in my python script. Refer to `helperFunction.py` to understand the process but I'm setting the minimum of 0 and maximum of 1 to help with some machine learning applications later on in this report. 


```python
breastCancerNorm = normalize_df(breastCancer)
```

Let's try the `.describe()` function again and you'll see that all variables have a maximum of 1 which means we did our process correctly. 


```python
breastCancerNorm.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave_points_mean</th>
      <th>symmetry_mean</th>
      <th>fractal_dimension_mean</th>
      <th>radius_se</th>
      <th>texture_se</th>
      <th>perimeter_se</th>
      <th>area_se</th>
      <th>smoothness_se</th>
      <th>compactness_se</th>
      <th>concavity_se</th>
      <th>concave_points_se</th>
      <th>symmetry_se</th>
      <th>fractal_dimension_se</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave_points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.372583</td>
      <td>0.338222</td>
      <td>0.323965</td>
      <td>0.332935</td>
      <td>0.216920</td>
      <td>0.394785</td>
      <td>0.260601</td>
      <td>0.208058</td>
      <td>0.243137</td>
      <td>0.379605</td>
      <td>0.270379</td>
      <td>0.106345</td>
      <td>0.189324</td>
      <td>0.099376</td>
      <td>0.062636</td>
      <td>0.181119</td>
      <td>0.174439</td>
      <td>0.080540</td>
      <td>0.223454</td>
      <td>0.178143</td>
      <td>0.100193</td>
      <td>0.296663</td>
      <td>0.363998</td>
      <td>0.283138</td>
      <td>0.170906</td>
      <td>0.404138</td>
      <td>0.220212</td>
      <td>0.217403</td>
      <td>0.393836</td>
      <td>0.263307</td>
      <td>0.189596</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.483918</td>
      <td>0.166787</td>
      <td>0.145453</td>
      <td>0.167915</td>
      <td>0.149274</td>
      <td>0.126967</td>
      <td>0.161992</td>
      <td>0.186785</td>
      <td>0.192857</td>
      <td>0.138456</td>
      <td>0.148702</td>
      <td>0.100421</td>
      <td>0.121917</td>
      <td>0.095267</td>
      <td>0.084967</td>
      <td>0.102067</td>
      <td>0.134498</td>
      <td>0.076227</td>
      <td>0.116884</td>
      <td>0.116316</td>
      <td>0.091417</td>
      <td>0.171940</td>
      <td>0.163813</td>
      <td>0.167352</td>
      <td>0.139932</td>
      <td>0.150779</td>
      <td>0.152649</td>
      <td>0.166633</td>
      <td>0.225884</td>
      <td>0.121954</td>
      <td>0.118466</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.223342</td>
      <td>0.218465</td>
      <td>0.216847</td>
      <td>0.117413</td>
      <td>0.304595</td>
      <td>0.139685</td>
      <td>0.069260</td>
      <td>0.100944</td>
      <td>0.282323</td>
      <td>0.163016</td>
      <td>0.043781</td>
      <td>0.104690</td>
      <td>0.040004</td>
      <td>0.020635</td>
      <td>0.117483</td>
      <td>0.081323</td>
      <td>0.038106</td>
      <td>0.144686</td>
      <td>0.102409</td>
      <td>0.046750</td>
      <td>0.180719</td>
      <td>0.241471</td>
      <td>0.167837</td>
      <td>0.081130</td>
      <td>0.300007</td>
      <td>0.116337</td>
      <td>0.091454</td>
      <td>0.223127</td>
      <td>0.185098</td>
      <td>0.107700</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.302381</td>
      <td>0.308759</td>
      <td>0.293345</td>
      <td>0.172895</td>
      <td>0.390358</td>
      <td>0.224679</td>
      <td>0.144189</td>
      <td>0.166501</td>
      <td>0.369697</td>
      <td>0.243892</td>
      <td>0.077023</td>
      <td>0.165267</td>
      <td>0.072092</td>
      <td>0.033112</td>
      <td>0.158650</td>
      <td>0.136675</td>
      <td>0.065379</td>
      <td>0.207047</td>
      <td>0.152643</td>
      <td>0.079191</td>
      <td>0.250445</td>
      <td>0.356876</td>
      <td>0.235320</td>
      <td>0.123206</td>
      <td>0.397081</td>
      <td>0.179110</td>
      <td>0.181070</td>
      <td>0.343402</td>
      <td>0.247782</td>
      <td>0.163977</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.416442</td>
      <td>0.408860</td>
      <td>0.416765</td>
      <td>0.271135</td>
      <td>0.475490</td>
      <td>0.340531</td>
      <td>0.306232</td>
      <td>0.367793</td>
      <td>0.453030</td>
      <td>0.340354</td>
      <td>0.133044</td>
      <td>0.246155</td>
      <td>0.122509</td>
      <td>0.071700</td>
      <td>0.218683</td>
      <td>0.226800</td>
      <td>0.106187</td>
      <td>0.278651</td>
      <td>0.219480</td>
      <td>0.126556</td>
      <td>0.386339</td>
      <td>0.471748</td>
      <td>0.373475</td>
      <td>0.220901</td>
      <td>0.494156</td>
      <td>0.302520</td>
      <td>0.305831</td>
      <td>0.554639</td>
      <td>0.318155</td>
      <td>0.242949</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Box Plot of Transformed Data

Now to further illustrate the transformation let's create a *boxplot* of the scaled data set, and see the difference from our first *boxplot*. 


```python
f, ax = plt.subplots(figsize=(11, 15))

ax.set_axis_bgcolor('#fafafa')
plt.title("Box Plot of Transformed Data Set (Breast Cancer Wisconsin Data Set)")
ax.set(xlim=(-.05, 1.05))
ax = sns.boxplot(data = breastCancerNorm[1:29], orient = 'h', palette = 'Set2')
```


![png](breastCancerMachineLearningPresentation_files/breastCancerMachineLearningPresentation_27_0.png)


# Model Estimation
## Creating Training and Test Sets

For this next process, we split the data set into our training and test sets which will be (pseudo) randomly selected having a 80-20% splt. 


```python
# Here we do a 80-20 split for our training and test set
train, test = train_test_split(breastCancer, 
                               test_size = 0.20, 
                               random_state = 42)

# Create the training test omitting the diagnosis
training_set = train.ix[:, train.columns != 'diagnosis']
# Next we create the class set (Called target in Python Documentation)
# Note: This was confusing af to figure out cus the documentation is low-key kind of shitty
class_set = train.ix[:, train.columns == 'diagnosis']

# Next we create the test set doing the same process as the training set
test_set = test.ix[:, test.columns != 'diagnosis']
test_class_set = test.ix[:, test.columns == 'diagnosis']
```

# Kth Nearest Neighbor

A popular algorithm within classification, *kth nearest neighbor* employs a majority votes methodology using *Euclidean Distance* (a.k.a straight-line distance between two points) based on the specificied *k*. 

So within context of my data set, I employ *k=9* so the algorithm looks for the 9 neighbors closest to the value its trying to classify (again closest measured using [Euclidean Distance](http://mathworld.wolfram.com/Distance.html). This algorithm is known as a *lazy algorithm* and is considered one of the simpler algorithms in *Machine Learning*. 


I like using *kth nearest neighbor* because it gives me a base to compare other methods. One can often find that *K-nn* will perform better than more complicated methods, which by *Occam's Razor* will signal you to choose this method even though other methods are seen as *cooler* and what not. 

**Important to Note**: The biggest drawback for *K-NN* is the [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)  

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




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=7, p=2,
               weights='uniform')



### Training Set Calculations
For this set, we calculate the accuracy based on the training set. I decided to include this to give context, but within *Machine Learning* processes we aren't concerned with the *training error rate* since it can suffer from high *bias* and over-fitting. 

This is why we create a test set to *test* the algorithm against data it hasn't seen before. This is important if you are just getting into *Machine Learning*. Of course *cross-validation* is more effective (which I will probably employ on a later iteration) in measuring the algorithm's performance, it is important to understand how *training* and *test sets* function.


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

print("Here is our accuracy for our training set: {0: .3f} ".format(accuracyTrain))
```

    Actual Values       0    1
    Predicted Values          
    0                 279   20
    1                   7  149
    Here is our accuracy for our training set:  0.941 



```python
train_error_rate = 1 - accuracyTrain   
print("The train error rate for our model is: {0: .3f}".format(train_error_rate))
```

    The train error rate for our model is:  0.059


## Cross Validation 
We do cross validation as another form of accuracy since there can be bias in our training and test set methodology. Setting different seeds and creating different training and test sets can lead to drastic differences in *Test Error Rate*. 


```python
n = KFold(n_splits=10)

scores = cross_val_score(fit_KNN, 
                         test_set, 
                         test_class_set['diagnosis'], 
                         cv = n)
print("Accuracy: {0: 0.2f} (+/- {1: 0.2f})"\
      .format(scores.mean(), scores.std() / 2))
```

    Accuracy:  0.95 (+/-  0.04)


### Test Set Evaluations

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
print("Here is our accuracy for our test set: {0: .3f}".format(accuracy))
```

    Actual Values      0   1
    Predicted Values        
    0                 70   4
    1                  1  39
    Here is our accuracy for our test set:  0.956



```python
# Here we calculate the test error rate!
test_error_rate = 1 - accuracy
print("The test error rate for our model is: {0: .3f}".format(test_error_rate))
```

    The test error rate for our model is:  0.044


### Conclusion for K-NN

So as you can see our **K-NN** model did pretty well! It had an even split of *false positives* and *false negatives* (2 for 2). Great this gives us a base to compare the upcoming models as well as to decide whether the other models are *over-kill* for this data set. 

### Calculating for later use in ROC Curves

Here we calculate the *false positive rate* and *true positive rate* for our model. This will be used later when creating **ROC Curves** when comparing our models. I will go into more detail in that section as well!. 


```python
fpr, tpr, _ = roc_curve(predictions, test_class_set)
```

###  Calculations for Area under the Curve 
This function calculates the area under the curve which you will see the relevancy later in this project, but ideally we want it to be closest to 1 as possible. 


```python
auc_knn = auc(fpr, tpr)
```

# Decision Trees

*Decision trees* have a hierarchical structure similar to asking a series of questions until we can deduce the classifications. Each leaf represents a class label while the branches represent the process the tree used to deduce the *leaf nodes* (class label). 

The structure is described as such:
+ The very first node is called the *root node*
+ the nodes within the tree are called the *internal nodes*
+ the nodes at the end of t

For this model we use an index known as the [Gini Impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity), its the default index, but I included in the model to drive the point home. 

**Important to Note**: Some of the documentation with respect to models in *sklearn* are still confusing for me so I was not able to understand how to go about pruning my tree. Thus to prevent over-fitting I made the `max_depth` equal to 3. And left it as such. A more indepth analysis would utilize this information to reduce dimensions using methods like *Principal Component Analysis*.


```python
dt = DecisionTreeClassifier(random_state = 42, 
                            criterion='gini', 
                            max_depth=3)

fit_dt = dt.fit(training_set, class_set)
```

Decision trees are important classification models, because often follow decision making similar to that of human decision making. Although it is important to state that they will often perform very poor compared to other predictive modeling. 

The next step we export the image representation of our *Decision Tree* path using `export_graphviz` as a *dot* file (Graph Description language) where we add the `feature_names` or else we get the outputs as follows (which isn't very helpful):

    X[N] where N is the index of the feature space
    
Since I don't have graphviz downloaded on my machine, I use this following online interpretter that will take the plain text from the *dot* file (which I opened with **Sublime Text** and copy pasted) and create the visual graph for free! (Source: https://github.com/dreampuf/GraphvizOnline)


```python
namesInd = names[2:] # Cus the name list has 'id_number' and 'diagnosis' so we exclude those

with open('breastCancerWD.dot', 'w') as f:
    f = export_graphviz(fit_dt, out_file = f,
                        feature_names=namesInd,
                        rounded = True)
```

<img src='dotFiles/dtWD.png'>

### Variable Importance

A useful feature within what are known as *CART* (Classification And Regression Trees) is extracting which features are important when using the *Gini Impurity*. For this next process we will be grabbing the index of these features, then using a `for loop` to state which were the most important. 


```python
importances = fit_dt.feature_importances_
indices = np.argsort(importances)[::-1]
```


```python
# Print the feature ranking
print("Feature ranking:")

for f in range(30):
    i = f
    print("%d. The feature '%s' has a Gini Importance of %f" % (f + 1, 
                                                                namesInd[indices[i]], 
                                                                importances[indices[f]]))
```

    Feature ranking:
    1. The feature 'concave_points_mean' has a Gini Importance of 0.752304
    2. The feature 'concave_points_worst' has a Gini Importance of 0.071432
    3. The feature 'radius_worst' has a Gini Importance of 0.056905
    4. The feature 'perimeter_worst' has a Gini Importance of 0.056028
    5. The feature 'texture_mean' has a Gini Importance of 0.030106
    6. The feature 'fractal_dimension_se' has a Gini Importance of 0.020188
    7. The feature 'area_se' has a Gini Importance of 0.013038
    8. The feature 'concavity_mean' has a Gini Importance of 0.000000
    9. The feature 'radius_se' has a Gini Importance of 0.000000
    10. The feature 'fractal_dimension_mean' has a Gini Importance of 0.000000
    11. The feature 'symmetry_mean' has a Gini Importance of 0.000000
    12. The feature 'fractal_dimension_worst' has a Gini Importance of 0.000000
    13. The feature 'texture_se' has a Gini Importance of 0.000000
    14. The feature 'smoothness_mean' has a Gini Importance of 0.000000
    15. The feature 'area_mean' has a Gini Importance of 0.000000
    16. The feature 'perimeter_mean' has a Gini Importance of 0.000000
    17. The feature 'compactness_mean' has a Gini Importance of 0.000000
    18. The feature 'smoothness_se' has a Gini Importance of 0.000000
    19. The feature 'perimeter_se' has a Gini Importance of 0.000000
    20. The feature 'symmetry_worst' has a Gini Importance of 0.000000
    21. The feature 'compactness_se' has a Gini Importance of 0.000000
    22. The feature 'concavity_se' has a Gini Importance of 0.000000
    23. The feature 'concave_points_se' has a Gini Importance of 0.000000
    24. The feature 'symmetry_se' has a Gini Importance of 0.000000
    25. The feature 'texture_worst' has a Gini Importance of 0.000000
    26. The feature 'area_worst' has a Gini Importance of 0.000000
    27. The feature 'smoothness_worst' has a Gini Importance of 0.000000
    28. The feature 'compactness_worst' has a Gini Importance of 0.000000
    29. The feature 'concavity_worst' has a Gini Importance of 0.000000
    30. The feature 'radius_mean' has a Gini Importance of 0.000000


As you can see here after the 7th feature which is `area_se`, the *Gini Importance* becomes 0 for all remaining variables. 

We cannot make statistical conclusions as to its significance because from prior knowledge I know that *Random Forest* will perform significantly better and give us more insight into the data set. But for now we will continue to do the calculations and receive the test error rates for this model. 
### Test Set Evaluations


```python
accuracy_dt = fit_dt.score(test_set, test_class_set['diagnosis'])

print("Here is our mean accuracy on the test set:\n {0: .3f}".format(accuracy_dt))
```

    Here is our mean accuracy on the test set:
      0.947



```python
predictions_dt = fit_dt.predict(test_set)

print("Table comparing actual vs. predicted values for our test set:\n",
     pd.crosstab(predictions_dt, test_class_set['diagnosis'], 
                  rownames=['Predicted Values'], 
                  colnames=['Actual Values']))
```

    Table comparing actual vs. predicted values for our test set:
     Actual Values      0   1
    Predicted Values        
    0                 69   4
    1                  2  39



```python
# Here we calculate the test error rate!
test_error_rate_dt = 1 - accuracy_dt

print("The test error rate for our model is:\n {0: .3f}".format(test_error_rate_dt))
```

    The test error rate for our model is:
      0.053


### Calculating for later use in ROC Curves


```python
fpr1, tpr1, _ = roc_curve(predictions_dt, test_class_set)
```

### Calculations for Area under Curve 


```python
auc_dt = auc(fpr1, tpr1)
```

### Conclusions for Decision Trees
As we can see there is still a lot to be desired in performing a succinct analysis using *Decision Trees*. Obviously *Cross-Validation* would be helpful in understanding what appropriate depth and other significant parameters should be done to optimize our model. For this iteration this will not be included, but I do plan on expanding on this in later iterations. A lot of my confusion when doing this project is the documentation leaves a lot to be desired, but with every iteration I'm learning more and more about the functions and their capabilies.

# Random Forest
Also known as *Random Decision Forest* is just that an entire forest of random decision trees. This is an extension of *Decision Trees* that will perform significantly better than a single tree because it corrects over-fitting. Here is a brief overview of the evolution of *CART* analysis, but the process of better evaluations goes as follows:
+ Single Decision Tree (Single tree)
+ Bagging Trees (Multiple trees) [Model with all features, M, considered at splits, where M = all features]
+ Random Forest (Multiple trees) [Model with m features considered at splits, where m < M]

### Bagging Trees
*Decision Trees* tend to have *low bias and high variance*, a process known as *Bagging Trees* (*Bagging* = *Bootstrap Aggregating*) was an extension that does random sampling with replacement where after creating N trees it classifies on majority votes. This process reduces the variance while at the same time keeping the bias low. However, a downside to this process is if certain features are strongly predictors then too many trees will employ these features causing correlation between the trees. 

Thus *Random Forest* aims to reduce this correlation by choosing only a subsample of the feature space at each split. Essentially aiming to make the trees more independent thereby reducing the variance.   

Generally, we aim to create 500 trees and use our m to be sqrt(M) rounded down. So since we have 30 features I will use 5 for my `max_features` parameter. Recall I will be using the *Gini Importance* metric, although it is the default I will always include it in this project to give context. 


```python
fit_RF = RandomForestClassifier(random_state = 42, 
                                criterion='gini',
                                n_estimators = 500,
                                max_features = 5)
```


```python
fit_RF.fit(training_set, class_set['diagnosis'])
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features=5, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=500, n_jobs=1, oob_score=False, random_state=42,
                verbose=0, warm_start=False)



## Variable Importance

Essentially the same process as the *Decision Trees*, but we gather this instance from **500** trees! 


```python
importancesRF = fit_RF.feature_importances_
indicesRF = np.argsort(importancesRF)[::-1]
indicesRF
```




    array([27, 23,  7, 22, 20,  6,  0,  2,  3, 26, 13, 21, 10,  1, 25, 28, 24,
            5, 12, 16,  4, 19, 29, 15, 18,  9, 17, 11,  8, 14])




```python
# Print the feature ranking
print("Feature ranking:")

for f in range(30):
    i = f
    print("%d. The feature '%s' has a Gini Importance of %f" % (f + 1, 
                                                                namesInd[indicesRF[i]], 
                                                                importancesRF[indicesRF[f]]))
```

    Feature ranking:
    1. The feature 'concave_points_worst' has a Gini Importance of 0.139713
    2. The feature 'area_worst' has a Gini Importance of 0.122448
    3. The feature 'concave_points_mean' has a Gini Importance of 0.115332
    4. The feature 'perimeter_worst' has a Gini Importance of 0.114410
    5. The feature 'radius_worst' has a Gini Importance of 0.082506
    6. The feature 'concavity_mean' has a Gini Importance of 0.051091
    7. The feature 'radius_mean' has a Gini Importance of 0.047065
    8. The feature 'perimeter_mean' has a Gini Importance of 0.041769
    9. The feature 'area_mean' has a Gini Importance of 0.040207
    10. The feature 'concavity_worst' has a Gini Importance of 0.038435
    11. The feature 'area_se' has a Gini Importance of 0.029797
    12. The feature 'texture_worst' has a Gini Importance of 0.021006
    13. The feature 'radius_se' has a Gini Importance of 0.016963
    14. The feature 'texture_mean' has a Gini Importance of 0.016359
    15. The feature 'compactness_worst' has a Gini Importance of 0.015939
    16. The feature 'symmetry_worst' has a Gini Importance of 0.013319
    17. The feature 'smoothness_worst' has a Gini Importance of 0.013109
    18. The feature 'compactness_mean' has a Gini Importance of 0.012102
    19. The feature 'perimeter_se' has a Gini Importance of 0.010395
    20. The feature 'concavity_se' has a Gini Importance of 0.007546
    21. The feature 'smoothness_mean' has a Gini Importance of 0.007518
    22. The feature 'fractal_dimension_se' has a Gini Importance of 0.006149
    23. The feature 'fractal_dimension_worst' has a Gini Importance of 0.005899
    24. The feature 'compactness_se' has a Gini Importance of 0.005324
    25. The feature 'symmetry_se' has a Gini Importance of 0.004980
    26. The feature 'fractal_dimension_mean' has a Gini Importance of 0.004723
    27. The feature 'concave_points_se' has a Gini Importance of 0.004516
    28. The feature 'texture_se' has a Gini Importance of 0.004405
    29. The feature 'symmetry_mean' has a Gini Importance of 0.003570
    30. The feature 'smoothness_se' has a Gini Importance of 0.003402


We don't run into the same issue as *Decision Trees* when looking at the *Variable Importance*, and we can be more sure that our *Random Forest* model gives us significant results for our analysis. Let's create a barplot showcasing the *Variable Importance* against the *Gini Importance*.

### Feature Importance Visual

Here I use the `sorted` function to sort the *Gini Importance* criterion from least to greatest which was a work around in order to create a horizontal barplot, as well as creating an index using the `arange` function in `numpy`


```python
indRf = sorted(importancesRF) # Sort by Decreasing order
index = np.arange(30)
index
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])




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


![png](breastCancerMachineLearningPresentation_files/breastCancerMachineLearningPresentation_74_0.png)



```python
# CROSS VALIDATION
n = KFold(n_splits=10)
scores = cross_val_score(fit_RF, 
                         test_set, 
                         test_class_set['diagnosis'], 
                         cv = n)

print("Accuracy: {0: 0.2f} (+/- {1: 0.2f})"\
      .format(scores.mean(), scores.std() / 2))
```

    Accuracy:  0.96 (+/-  0.03)


## Test Set Evaluations


```python
predictions_RF = fit_RF.predict(test_set)
```


```python
print(pd.crosstab(predictions_RF, 
                  test_class_set['diagnosis'], 
                  rownames=['Predicted Values'], 
                  colnames=['Actual Values']))
```

    Actual Values      0   1
    Predicted Values        
    0                 70   3
    1                  1  40



```python
accuracy_RF = fit_RF.score(test_set, test_class_set['diagnosis'])

print("Here is our mean accuracy on the test set:\n {0:.3f}"\
      .format(accuracy_RF))
```

    Here is our mean accuracy on the test set:
     0.965



```python
# Here we calculate the test error rate!
test_error_rate_RF = 1 - accuracy_RF
print("The test error rate for our model is:\n {0: .3f}"\
      .format(test_error_rate_RF))
```

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
Edit later 

# Neural Network

First we have to scale our data. 

## Normalzing Data


```python
breastCancerNorm = normalize_df(breastCancer)
```


```python
training_set_norm, class_set_norm, test_set_norm, test_class_set_norm = splitSets(breastCancerNorm)
```


```python
fit_NN = MLPClassifier(solver='lbfgs', 
                       hidden_layer_sizes=(5, ), 
                       activation='logistic',
                       random_state=7)
```


```python
fit_NN.fit(training_set_norm, 
           class_set_norm['diagnosis'])
```




    MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(5,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=7, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)




```python
n = KFold(n_splits=10)
scores = cross_val_score(fit_NN, 
                         test_set_norm, 
                         test_class_set_norm['diagnosis'], 
                         cv = n)

print("Accuracy: {0: 0.2f} (+/- {1: 0.2f})"\
      .format(scores.mean(), scores.std() / 2))
```

    Accuracy:  0.93 (+/-  0.03)


Left here to show that *Neural Networks* perform poorly when appropriate pre-processing transformations *aren't* made. 


```python
predictions_NN = fit_NN.predict(test_set_norm)

print(pd.crosstab(predictions_NN, test_class_set_norm['diagnosis'], 
                  rownames=['Predicted Values'], 
                  colnames=['Actual Values']))
```

    Actual Values      0   1
    Predicted Values        
    0                 69   1
    1                  2  42



```python
accuracy_NN = fit_NN.score(test_set_norm, test_class_set_norm['diagnosis'])

print("Here is our mean accuracy on the test set:\n{0: .2f} ".format(accuracy_NN))
```

    Here is our mean accuracy on the test set:
     0.97 



```python
# Here we calculate the test error rate!
test_error_rate_NN = 1 - accuracy_NN
print("The test error rate for our model is:\n {0: .3f}"\
      .format(test_error_rate_NN))
```

    The test error rate for our model is:
      0.026



```python
fpr3, tpr3, _ = roc_curve(predictions_NN, test_class_set)
```


```python
auc_nn = auc(fpr3, tpr3)
```

# ROC Curves

*Receiver Operating Characteristc* Curve calculations we did using the function `roc_curve` were calculating the **False Positive Rates** and **True Positive Rates** for each model. We will now graph these calculations, and being located the top left corner of the plot indicates a really ideal model, i.e. a **False Positive Rate** of 0 and **True Positive Rate** of 1, so we plot the *ROC Curves* for all our models in the same axis and see how they compare!

We also calculated the **Area under the Curve**, so our curve in this case are the *ROC Curves*, we then place these calculations in the legend with their respective models. 


```python
f, ax = plt.subplots(figsize=(15, 15))

plt.plot(fpr, tpr, label='Kth-NN ROC Curve (area = {0: .4f})'.format(auc_knn), 
         color = 'deeppink', 
         linewidth=1)
plt.plot(fpr1, tpr1,label='Decision Trees ROC Curve (area = {0: .4f})'.format(auc_dt), 
         color = 'navy', 
         linewidth=2)
plt.plot(fpr2, tpr2,label='Random Forest ROC Curve (area = {0: .4f})'.format(auc_rf), 
         color = 'red', 
         linestyle=':', 
         linewidth=2)
plt.plot(fpr3, tpr3,label='Neural Networks ROC Curve (area = {0: .4f})'.format(auc_nn), 
         color = 'aqua', 
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


![png](breastCancerMachineLearningPresentation_files/breastCancerMachineLearningPresentation_99_0.png)


Let's zoom in to get a better picture!

### ROC Curve Plot Zoomed in


```python
f, ax = plt.subplots(figsize=(15, 15))
plt.plot(fpr, tpr, label='Kth-NN ROC Curve  (area = {0: .4f})'.format(auc_knn), 
         color = 'deeppink', 
         linewidth=1)
plt.plot(fpr1, tpr1,label='Decision Trees ROC Curve  (area = {0: .4f})'.format(auc_dt), 
         color = 'navy', 
         linewidth=2)
plt.plot(fpr2, tpr2,label='Random Forest ROC Curve  (area = {0: .4f})'.format(auc_rf), 
         color = 'red', 
         linestyle=':', 
         linewidth=3)
plt.plot(fpr3, tpr3,label='Neural Networks ROC Curve  (area = {0: .4f})'.format(auc_nn), 
         color = 'aqua', 
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


![png](breastCancerMachineLearningPresentation_files/breastCancerMachineLearningPresentation_101_0.png)


From the `auc` calculations we can see that both *Random Forest* and *Neural Networks* performed better than *Kth Nearest Neighbor* and *Decision Trees* which is pretty intuitive. 

Also visually examining the plot, *Random Forest* is noticeable more elevated than the other models which is indicative of a good prediction tool, using this form of diagnositcs. I will go into more detail later on, but for now this will do. 

## Conclusions
Once I employed all these methods, we can that **Neural Networks** performed the best in terms of most diagnostics. *Kth Nearest Neighbor* performed better in terms of *cross validation*, but I have yet to perform *hyperparameter optimization* on other processes.  This project is an iterative process, so I will be working to reach a final consensus. In terms of most insight into the data, *random forest* model is able to tell us the most of our model. In terms of cross validated performance *kth nearest neighbor* performed the best.  

### Diagnostics for Data Set

| Model/Algorithm 	| Test Error Rate 	| False Negative for Test Set 	| Area under the Curve for ROC | Cross Validation Score | 
|-----------------|-----------------|-------------------------------|----------------------------|-----------|
| Kth Nearest Neighbor* | 0.035 |	2 |	0.963 | 0.966 (+/-  0.021) | 
| Decision Trees 	| 0.053 	| 4 |	0.948 | 0.920 (+/-  0.024) | 
| Random Forest 	|  0.035	| 3 	| 0.9673 |  0.955 (+/-  0.030) |  
| Neural Networks 	| 0.026 	| 1 	| 0.981 | 0.930 (+/-  0.034) |  

\*Only model with *Hyperparameter optimization* done
