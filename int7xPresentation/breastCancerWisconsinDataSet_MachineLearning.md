# Abstract
<img src="https://www.researchgate.net/profile/Syed_Ali39/publication/41810238/figure/fig5/AS:281736006127621@1444182506838/Figure-2-Fine-needle-aspiration-of-a-malignant-solitary-fibrous-tumor-is-shown-A-A.png">


Ex. Image of a malignant solitary fibrous tumor using **FNA**

This is popular data set used for machine learning purposes, and I plan on using the same techniques I used for another data set that performed poorly due to having too many *Categorical* variables (**NOTE**: Learned about *dummy variables* so might revisit and execute analysis correctly this time!)

Here are the **Machine learning methods** I decided to use:

+ Kth Nearest Neighbor
+ Decision Trees
+ (Bagging) Random Forest
+ Neural Networks

I employ critical data analysis modules in this project, emphasizing on: 

+ pandas
+ scikit learn 
+ matplotlib (for visuals)
+ seaborn (easier to make statistical plots)

We load our modules into our python environment. In my case I am employing a **Jupyter Notebook** while running inside an **conda** environment. 

For now to illustrate and show the module versions in a simple way I will name the ones I used and show the version I used as follows:

+ numpy==1.11.2
+ pandas==0.18.1
+ matplotlib==1.5.3
+ sklearn==0.18.1
+ seaborn=0.7.1


<!-- BODY -->
<!-- LOAD MODULES -->
# Load Modules
We load our modules into our python environment. In my case I am employing a **Jupyter Notebook** while running inside an **|conda** environment. 

For now to illustrate and show the module versions in a simple way I will name the ones I used and show the version I used as follows:

+ numpy==1.11.2
+ pandas==0.18.1
+ matplotlib==1.5.3
+ sklearn==0.18.1
+ seaborn=0.7.1

Should include the *shebang*:

	#!/usr/bin/env python3
	import numpy as np
	import pandas as pd # Data frames
	import matplotlib.pyplot as plt # Visuals
	import seaborn as sns # Danker visuals
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

## Loading Data
For this section, I'll load the data into a **Pandas** dataframe using `urlopen` from the `urllib.request` module. 

Instead of downloading a csv, I started implementing this method(Inspired by Jason's Python Tutorials) where I grab the data straight from the **UCI Machine Learning Database**. Makes it easier to go about analysis from online sources and cuts out the need to download/upload a csv file when uploading on **GitHub**. I create a list with the appropriate names and set them within the dataframe. **NOTE**: The names were not documented to well so I used [this analysis](https://www.kaggle.com/buddhiniw/d/uciml/breast-cancer-wisconsin-data/breast-cancer-prediction) to grab the variable names and some other tricks that I didn't know that were available in **Python** (I will mention the use in the script!)

Finally I set the column `id_number` as the index for the dataframe. 


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


### Outputting our data
Its good to always output sections of your data so you can give context to the reader as to what each column looks like, as well as seeing examples of how the data is suppose to be formatted when loaded correctly. Many people run into the issue (especially if you run into a data set with poor documentation w.r.t. the column names), so its good habit to show your data during your analysis. 

We use the function `head()` which is essentially the same as the `head` function in **R** if you come from an **R** background. Notice the syntax for **Python** is significantly different than that of **R** though. 

	breastCancer.head()

<!-- MISSING OUTPUT -->

### More Preliminary Analysis
Much of these sections are given to give someone context to the dataset you are utilizing. Often looking at raw data will not give people the desired context, so it is important for us as data enthusiast to fill in the gaps for people who are interested in the analysis. But don't plan on running it anytime soon. 

#### Data Frame Dimensions
Here we use the `.shape` function to give us the lengths of our data frame, where the first output is the row-length and the second output is the column-length. 

#### Data Types
Another important piece of information that is **important** is the data types of your variables in the data set. 

It is often good practice to check the variable types with either the source or with your own knowledge of the data set. For this data set, we know that all variables are measurements, so they are all continous (Except **Dx**), so no further processing is needed for this step.

An common error that can happen is: if a variable is *discrete* (or *categorical*), but has a numerical representation someone can easily forget the pre-processing and do analysis on the data type as is. Since they are numeric they will be interpretted as either `int` or `float` which is incorrect. I can go on and on, but for this data set the numerical representation of the **Dx** is correct and is referred to as *indicator* or *dummy* variables. 

	print("Here's the dimensions of our data frame:\n", 
	breastCancer.shape)
	print("Here's the data types of our columns:\n",
	breastCancer.dtypes)

## Terminal Output

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

	# Converted to binary to help later on with models and plots
	breastCancer['diagnosis'] = breastCancer['diagnosis'].map({'M':1, 'B':0})
	
	# Let's look at the count of the new representations of our Dx's
	breastCancer['diagnosis'].value_counts()

## Class Imbalance
The count for our Dx is important because it brings up the discussion of *Class Imbalance* within *Machine learning* and *data mining* applications. 

*Class Imbalance* refers to when a class within a data set is outnumbered by the other class (or classes). 
Reading documentation online, *Class Imbalance* is present when a class populates 10-20% of the data set. 

However for this data set, its pretty obvious that we don't suffer from this, but since I'm practicing my **Python**, I decided to experiment with `functions` to get better at **Python**!

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

Let's check if this worked. I'm sure there's more effective ways of doing this process, but this is me doing brute-force attempts to defining/creating working functions. Don't worry I'll get better with practice :)

	classImbalance('diagnosis')

## Terminal Output
	
	The percentage of Malignant Dx is: 37.26%
	The percentage of Begnin Dx is: 62.74%

As we can see here our data set is not suffering from *class imbalance* so we can proceed with our analysis. 

# Exploratory Analysis

An important process in **Machine Learning** is doing **Exploratory Analysis** to get a *feel* for your data. As well as creating visuals that can be digestable for anyone of any skill level. 

So I started by using the `.describe()` function to give some basic statistics relating to each variable. We can see there are 569 instances of each variable (which should make sense), but important to note that the distributions of the different variables have very high variance by looking at the **means** (Some can go as low as .0n while some as large as 800!)

	breastCancer.describe()

<!-- MISSING OUTPUT -->

We will discuss the high variance in the distribution of the variables later within context of the appropriate analysis. For now we move on to visual representations of our data. Still a continuation of our **Exploratory Analysis**.

# Visual Exploratory Analysis
For this section we utilize the module `Seaborn` which contains many powerful statistical graphs that would have been hard to produce using `matplotlib` (My note: `matplotlib` is not the most user friendly like `ggplot2` in **R**, but **Python** seems like its well on its way to creating visually pleasing and inuitive plots!)

## Scatterplot Matrix
For this visual I cheated by referencing some variables that were indicators of being influencial to the analysis (See **Random Forest** Section).   


	# Variables chosen from Random Forest modeling. 
	breastCancerSamp = breastCancer.loc[:, 
                                    	['concave_points_worst', 'concavity_mean', 
                                     	'perimeter_worst', 'radius_worst', 
                                     	'area_worst', 'diagnosis']]
	
	sns.set_palette(palette = ('Red', '#875FDB'))
	pairPlots = sns.pairplot(breastCancerSamp, hue = 'diagnosis')
	pairPlots.set(axis_bgcolor='#fafafa')
	
	plt.show()

![]('images/breastCancerWisconsinDataSet_MachineLearning_19_0.png')

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

	corr = breastCancer.corr(method = 'pearson') # Correlation Matrix
	
	f, ax = plt.subplots(figsize=(11, 9))
	
	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(10, 275, as_cmap=True)
	
	
	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(corr,  cmap=cmap,square=True, 
            	xticklabels=True, yticklabels=True,
            	linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

![]('images/breastCancerWisconsinDataSet_MachineLearning_22_1.png')

We can see that our data set contains mostly positive correlation, as well as re-iterating to us that the 5 dependent variables we featured in the *Scatterplot Matrix* have strong *correlation*. Our variables don't have too much correlation so I won't go about doing feature extraction processes like **PCA**, but you are more welcomed to do so (you will probably get better prediction estimates).   

## Box Plot 

Don't think I need to go into too much explanation for *box plots* but I employed `Seaborn` on the entire data set. 

Remember our data is in its raw form, so the boxplots for now won't give us a good visual aid, but I included it to show the difference between our data set now and when it gets processed in a later step.

Recall on when using the `.describe()` function there were some variables that had a maximum value as high as **4k** while others had a minimum value as small as *.00n*.

So I had to set a limit on the x-axis to give a better picture of some of the variables. But basically its impossible to visualize them all given the high variance in distribution of our varibles.


	f, ax = plt.subplots(figsize=(11, 15))
	
	ax.set_axis_bgcolor('#fafafa')
	ax.set(xlim=(-.05, 50))
	plt.ylabel('Dependent Variables')
	plt.title("Box Plot of Pre-Processed Data Set")
	ax = sns.boxplot(data = breastCancer, orient = 'h', palette = 'Set2')

![]('images/breastCancerWisconsinDataSet_MachineLearning_25_0.png')

Not the best picture but this is a good segue into the next step in our *Machine learning* process.

## Normalizing data set

A step super important in *Machine Learning* is the known as *pre-processing*. 

What I will be doing in the next step is scaling the variables to fall within a specific range, more specifically they will fall between 0 and 1. As you have seen in the *Exploratory Analysis*, our data set had highly varying distributions. Although there is much literature discussing the scaling of variables, I found this observation relevant while researching for this project:

'*...by standardizing one attempts to give all variables an equal weight, in the hope of achieving objectivity. As such, it may be used by a practitioner who possesses no prior knowledge.*' - [Kaufman and Rousseeuw](https://www.amazon.com/dp/0471735787/?tag=stackoverfl08-20) 

Which as you may have guessed I am in no why an expert in breast cancer research so I thought this was a good justification for my analysis. 

Important to note that this excerpt was found and presented by StackOverflow user's [Franck Dernoncourt](http://stats.stackexchange.com/questions/41704/how-and-why-do-normalization-and-feature-scaling-work) response on the link hyperlinked to his username. 


	breastCancerFloat = breastCancer.iloc[:, 1:]
	
	for item in breastCancer:
    	if item in breastCancerFloat:
        	breastCancer[item] = ((breastCancer[item] - breastCancer[item].min()) / 
                              	(breastCancer[item].max() - breastCancer[item].min()))
	
	breastCancer.head()

<!-- MISSING OUTPUT -->

Let's try the `.describe()` function again and you'll see that all variables have a maximum of 1 which means we did our process correctly. 

	breastCancer.describe()

<!-- MISSING OUTPUT -->

## Box Plot of Transformed Data

Now to further illustrate the transformation let's create a *boxplot* of the scaled data set, and see the difference from our first *boxplot*. 

	f, ax = plt.subplots(figsize=(11, 15))
	
	ax.set_axis_bgcolor('#fafafa')
	plt.title("Box Plot of Transformed Data Set (Breast Cancer Wisconsin Data Set)")
	ax.set(xlim=(-.05, 1.05))
	ax = sns.boxplot(data = breastCancer[1:29], orient = 'h', palette = 'Set2')


![]('images/breastCancerWisconsinDataSet_MachineLearning_34_0.png')

From this visual we can see many variables have outliers as well some variables having very spread out data points like `compactness_worst` and `fractal_dimension_mean`. This can be important later on when diagnosising if outliers had a significant effect on our models. But for this iteration, that will not be included. 

# Model Estimation
## Creating Training and Test Sets

For this next process, we split the data set into our training and test sets which will be (pseudo) randomly selected having a 80-20% splt. 


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

# Kth Nearest Neighbor

A popular algorithm within classification, *kth nearest neighbor* employs a majority votes methodology using *Euclidean Distance* (a.k.a straight-line distance between two points) based on the specificied *k*. 

So within context of my data set, I employ *k=9* so the algorithm looks for the 9 neighbors closest to the value its trying to classify (again closest measured using [Euclidean Distance](http://mathworld.wolfram.com/Distance.html). This algorithm is known as a *lazy algorithm* and is considered one of the simpler algorithms in *Machine Learning*. 


I like using *kth nearest neighbor* because it gives me a base to compare other methods. One can often find that *K-nn* will perform better than more complicated methods, which by *Occam's Razor* will signal you to choose this method even though other methods are seen as *cooler* and what not. 

**Important to Note**: The biggest drawback for *K-NN* is the [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)  

### Set up the Algorithm
I will be employing `scikit-learn` algorithms for this process of my project, but in the future I would be interested in building the algorithms from "scratch". I'm not there yet with my learning though. 


	breastCancerKnn = KNeighborsClassifier(n_neighbors=9)

### Train the Model

Next we train the model on our *training set* and the *class set*. The other models follow a similar pattern, so its worth noting here I had to call up the `'diagnosis'` column in my `class_set` otherwise I got an error code. 


	breastCancerKnn.fit(training_set, class_set['diagnosis'])

## Terminal Output

	KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=9, p=2,
           weights='uniform')

## Training Set Calculations
For this set, we calculate the accuracy based on the training set. I decided to include this to give context, but within *Machine Learning* processes we aren't concerned with the *training error rate* since it can suffer from high *bias* and over-fitting. 

This is why we create a test set to *test* the algorithm against data it hasn't seen before. This is important if you are just getting into *Machine Learning*. Of course *cross-validation* is more effective (which I will probably employ on a later iteration) in measuring the algorithm's performance, it is important to understand how *training* and *test sets* function.

	# We predict the class for our training set
	predictionsTrain = breastCancerKnn.predict(training_set) 
	
	# Here we create a matrix comparing the actual values vs. the predicted values
	print(pd.crosstab(predictionsTrain, class_set['diagnosis'], 
                  	rownames=['Predicted Values'], colnames=['Actual Values']))
	
	# Measure the accuracy based on the trianing set
	accuracyTrain = breastCancerKnn.score(training_set, class_set['diagnosis'])
	
	print("Here is our accuracy for our training set:\n",
	'%.3f' % (accuracyTrain * 100), '%')


## Terminal Output

	Actual Values       0    1
	Predicted Values          
	0                 285    9
	1                   1  160
	Here is our accuracy for our training set:
	97.802 %


Now let's get the **training error rate**


	train_error_rate = 1 - accuracyTrain  
	print("The train error rate for our model is:\n",
	'%.3f' % (train_error_rate * 100), '%')

## Terminal Output


	The train error rate for our model is:
	2.198 %

## Test Set Evaluations

Next we start doing calculations on our *test set*, this will be our form of measurement that will indicate whether our model truly will perform as well as it did for our *training set*. 

We could have simply over-fitted our model, but since the model hasn't seen the *test set* it will be an un-biased measurement of accuracy. 

	# First we predict the Dx for the test set and call it predictions
	predictions = breastCancerKnn.predict(test_set)
	
	# Let's compare the predictions vs. the actual values
	print(pd.crosstab(predictions, test_class_set['diagnosis'], 
                  	rownames=['Predicted Values'], 
                  	colnames=['Actual Values']))
	
	# Let's get the accuracy of our test set
	accuracy = breastCancerKnn.score(test_set, test_class_set['diagnosis'])
	
	# TEST ERROR RATE!!
	print("Here is our accuracy for our test set:\n",
	'%.3f' % (accuracy * 100), '%')

## Terminal Output

	Actual Values      0   1
	Predicted Values        
	0                 69   2
	1                  2  41
	Here is our accuracy for our test set:
	96.491 %

	# Here we calculate the test error rate!
	test_error_rate = 1 - accuracy
	print("The test error rate for our model is:\n", 
	'%.3f' % (test_error_rate * 100), '%')


## Conclusion for K-NN

So as you can see our **K-NN** model did pretty well! It had an even split of *false positives* and *false negatives* (2 for 2). Great this gives us a base to compare the upcoming models as well as to decide whether the other models are *over-kill* for this data set. 

### Calculating for later use in ROC Curves

Here we calculate the *false positive rate* and *true positive rate* for our model. This will be used later when creating **ROC Curves** when comparing our models. I will go into more detail in that section as well!. 


	fpr, tpr, _ = roc_curve(predictions, test_class_set)

###  Calculations for Area under the Curve 
This function calculates the area under the curve which you will see the relevancy later in this project, but ideally we want it to be closest to 1 as possible. 

	auc_knn = auc(fpr, tpr)
