# Abstract

For this project, I wanted to implement a few **Machine Learning** techniques on a data set containing descriptive attributes of digitized images of a process known as, fine needle aspirate (**FNA**) of breast mass. We have a total of 29 features that were computed for each cell nucleus with an ID Number and the Diagnosis (Later converted to binary representations: **Malignant** = 1, **Benign** = 0). 

I used the same models as the other notebook, but this is the expanded data set, and goes more in-depth with the explanations for this project!
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

<img src="images/breastCancerWisconsinDataSet_MachineLearning_19_0.png" style="width: 100px;"/>

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

<img src="images/breastCancerWisconsinDataSet_MachineLearning_22_1.png" style="width: 100px;"/>

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

<img src="images/breastCancerWisconsinDataSet_MachineLearning_25_0.png" style="width: 100px;"/>

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


<img src="images/breastCancerWisconsinDataSet_MachineLearning_34_0.png" style="width: 100px;"/>

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


# Decision Trees

*Decision trees* have a hierarchical structure similar to asking a series of questions until we can deduce the classifications. Each leaf represents a class label while the branches represent the process the tree used to deduce the *leaf nodes* (class label). 

The structure is described as such:
+ The very first node is called the *root node*
+ the nodes within the tree are called the *internal nodes*
+ the nodes at the end of t

For this model we use an index known as the [Gini Impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity), its the default index, but I included in the model to drive the point home. 

**Important to Note**: Some of the documentation with respect to models in *sklearn* are still confusing for me so I was not able to understand how to go about pruning my tree. Thus to prevent over-fitting I made the `max_depth` equal to 3. And left it as such. A more indepth analysis would utilize this information to reduce dimensions using methods like *Principal Component Analysis*.


	dt = DecisionTreeClassifier(random_state = 42, 
                            	criterion='gini', 
                            	max_depth=3)
	
	fit = dt.fit(training_set, class_set)


Decision trees are important classification models, because often follow decision making similar to that of human decision making. Although it is important to state that they will often perform very poor compared to other predictive modeling. 

The next step we export the image representation of our *Decision Tree* path using `export_graphviz` as a *dot* file (Graph Description language) where we add the `feature_names` or else we get the outputs as follows (which isn't very helpful):

    X[N] where N is the index of the feature space
    
Since I don't have graphviz downloaded on my machine, I use this following online interpretter that will take the plain text from the *dot* file (which I opened with **Sublime Text** and copy pasted) and create the visual graph for free! (Source: https://github.com/dreampuf/GraphvizOnline)

	namesInd = names[2:] # Cus the name list has 'id_number' and 'diagnosis' so we exclude those
	
	with open('breastCancerWD.dot', 'w') as f:
    	f = export_graphviz(fit, out_file = f,
                        	feature_names=namesInd,
                        	rounded = True)

Image produced by **.dot** file:

<img src="images/dtWD.png" style="width: 100px;"/>

## Variable Importance

A useful feature within what are known as *CART* (Classification And Regression Trees) is extracting which features are important when using the *Gini Impurity*. For this next process we will be grabbing the index of these features, then using a `for loop` to state which were the most important. 

	importances = fit.feature_importances_
	indices = np.argsort(importances)[::-1]

Here's the implementation of the `for loop`:

	# Print the feature ranking
	print("Feature ranking:")
	
	for f in range(30):
    	i = f
    	print("%d. The feature '%s' has a Gini Importance of %f" % (f + 1, 
                                                                	namesInd[indices[i]], 
                                                                	importances[indices[f]]))

## Terminal Output

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


## Test Set Evaluations
The **test set evaluations** follow more or less the same processes as described earlier when using **Kth Nearest Neighbor** so I won't go into too much detail with explanations.

	accuracy_dt = fit.score(test_set, test_class_set['diagnosis'])
	
	print("Here is our mean accuracy on the test set:\n",
     	'%.2f' % (accuracy_dt * 100), '%')

## Terminal Output

	Here is our mean accuracy on the test set:
	94.74 %

Now let's compare the Actual Values against the Predicted Values

	predictions_dt = fit.predict(test_set)
	
	print("Table comparing actual vs. predicted values for our test set:\n",
     	pd.crosstab(predictions_dt, test_class_set['diagnosis'], 
                  	rownames=['Predicted Values'], 
                  	colnames=['Actual Values']))

## Terminal Output

	Table comparing actual vs. predicted values for our test set:
	Actual Values      0   1
	Predicted Values        
	0                 69   4
	1                  2  39


	# Here we calculate the test error rate!
	test_error_rate_dt = 1 - accuracy_dt
	print("The test error rate for our model is:\n",
		'%.3f' % (test_error_rate_dt * 100) , '%')

## Terminal Output

	The test error rate for our model is:
	5.263 %

### Calculating for later use in ROC Curves

	fpr1, tpr1, _ = roc_curve(predictions_dt, test_class_set)

### Calculations for Area under Curve 

	auc_dt = auc(fpr1, tpr1)

## Conclusions for Decision Trees
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


	fit_RF = RandomForestClassifier(random_state = 42, 
                                	criterion='gini',
                                	n_estimators = 500,
                                	max_features = 5)


	fit_RF.fit(training_set, class_set['diagnosis'])

## Terminal Output

	RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            	max_depth=None, max_features=5, max_leaf_nodes=None,
            	min_impurity_split=1e-07, min_samples_leaf=1,
            	min_samples_split=2, min_weight_fraction_leaf=0.0,
            	n_estimators=500, n_jobs=1, oob_score=False, random_state=42,
            	verbose=0, warm_start=False)


## Variable Importance

Essentially the same process as the *Decision Trees*, but we gather this instance from **500** trees! 

	importancesRF = fit_RF.feature_importances_
	indicesRF = np.argsort(importancesRF)[::-1]

Basically same process as **Decision tree** section

	# Print the feature ranking
	print("Feature ranking:")
	
	for f in range(30):
    	i = f
    	print("%d. The feature '%s' has a Gini Importance of %f" % (f + 1, 
                                                                	namesInd[indicesRF[i]], 
                                                                	importancesRF[indicesRF[f]]))


## Terminal Output

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


	indRf = sorted(importancesRF) # Sort by Decreasing order
	index = np.arange(30)

Now let's plot it. 

	f, ax = plt.subplots(figsize=(11, 11))
	
	ax.set_axis_bgcolor('#fafafa')
	plt.title('Feature importances for Random Forest Model')
	plt.barh(index, indRf,
        	align="center", 
        	color = '#875FDB')
	plt.yticks(index, ('smoothness_se', 'symmetry_mean', 'texture_se', 
                   	'concave_points_se', 'fractal_dimension_mean', 
                   	'symmetry_se', 'compactness_se', 'fractal_dimension_worst', 
                   	'fractal_dimension_se', 'smoothness_mean', 'concavity_se', 
                   	'perimeter_se', 'compactness_mean', 'smoothness_worst', 'symmetry_worst', 
                   	'compactness_worst', 'texture_mean', 'radius_se', 'texture_worst', 'area_se', 
                   	'concavity_worst', 'area_mean', 'perimeter_mean', 'radius_mean', 'concavity_mean', 
                   	'radius_worst', 'perimeter_worst', 'concave_points_mean', 
                   	'area_worst', 'concave_points_worst'))
	plt.ylim(-1, 30)
	plt.xlim(0, 0.15)
	plt.xlabel('Gini Importance')
	plt.ylabel('Feature')
	
	plt.show()

<img src="images/varImpRF.png" style="width: 100px;"/>

## Test Set Evaluations

	predictions_RF = fit_RF.predict(test_set)
	print(pd.crosstab(predictions_RF, test_class_set['diagnosis'], 
                  rownames=['Predicted Values'], 
                  colnames=['Actual Values']))

## Terminal Output

	Actual Values      0   1
	Predicted Values        
	0                 70   3
	1                  1  40

Let's get accuracy measurements like before

	accuracy_RF = fit_RF.score(test_set, test_class_set['diagnosis'])
	
	print("Here is our mean accuracy on the test set:\n",
     	'%.3f' % (accuracy_RF * 100), '%')

## Terminal Output

	Here is our mean accuracy on the test set:
	96.491 %

Test error rate!

	# Here we calculate the test error rate!
	test_error_rate_RF = 1 - accuracy_RF
	print("The test error rate for our model is:\n",
		'%.3f' % (test_error_rate_RF * 100), '%')

## Terminal Output

	The test error rate for our model is:
	3.509 %

### Calculating for later use in ROC Curves

	fpr2, tpr2, _ = roc_curve(predictions_RF, test_class_set)

### Calculations for  Area under Curve

	auc_rf = auc(fpr2, tpr2)


## Conclusions for Random Forest
Our *Random Forest* model performed significantly better than our *Decision Tree* as expected. Surprisingly *Kth Nearest Neighbor* had a lower test error rate, but overall the *Random Forest* model performed better and gave us more insight, especially when considering the amount of test subjects that were predicted *Begnin*, but were actually *Malignant*. This will be a strong factor when considering what model is the most appropriate for my analysis. 


# Neural Network

More on this method later. 

	fit_NN = MLPClassifier(solver='lbfgs', 
                       	hidden_layer_sizes=(5, ), 
                       	random_state=7)

Train the model

	fit_NN.fit(training_set, class_set['diagnosis'])

## Terminal Output

	MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       	beta_2=0.999, early_stopping=False, epsilon=1e-08,
       	hidden_layer_sizes=(5,), learning_rate='constant',
       	learning_rate_init=0.001, max_iter=200, momentum=0.9,
       	nesterovs_momentum=True, power_t=0.5, random_state=7, shuffle=True,
       	solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       	warm_start=False)

### Test Set Evaluations

	predictions_NN = fit_NN.predict(test_set)
	
	print(pd.crosstab(predictions_NN, test_class_set['diagnosis'], 
                  	rownames=['Predicted Values'], 
                  	colnames=['Actual Values']))

## Terminal Output

	Actual Values      0   1
	Predicted Values        
	0                 69   1
	1                  2  42

Same as before

	accuracy_NN = fit_NN.score(test_set, test_class_set['diagnosis'])
	
	print("Here is our mean accuracy on the test set:\n",
		'%.2f' % (accuracy_NN * 100), '%')

## Terminal Output

	Here is our mean accuracy on the test set:
	97.37 %

Yup 

	# Here we calculate the test error rate!
	test_error_rate_NN = 1 - accuracy_NN
	print("The test error rate for our model is:\n",
		'%.3f' % (test_error_rate_NN * 100), '%')

## Terminal Output

	The test error rate for our model is:
	2.632 %

### Calculating for later use in ROC Curves

	fpr3, tpr3, _ = roc_curve(predictions_NN, test_class_set)

### Calculations for Area under Curve 

	auc_nn = auc(fpr3, tpr3)

# ROC Curves

*Receiver Operating Characteristc* Curve calculations we did using the function `roc_curve` were calculating the **False Positive Rates** and **True Positive Rates** for each model. We will now graph these calculations, and being located the top left corner of the plot indicates a really ideal model, i.e. a **False Positive Rate** of 0 and **True Positive Rate** of 1, so we plot the *ROC Curves* for all our models in the same axis and see how they compare!

We also calculated the **Area under the Curve**, so our curve in this case are the *ROC Curves*, we then place these calculations in the legend with their respective models. 


	fig, ax = plt.subplots(figsize=(10, 10))
	
	plt.plot(fpr, tpr, label='Kth-NN ROC Curve (area = %.4f)' % auc_knn, 
         	color = 'deeppink', 
         	linewidth=1)
	plt.plot(fpr1, tpr1,label='Decision Trees ROC Curve (area = %.4f)' % auc_dt, 
         	color = 'navy', 
         	linewidth=2)
	plt.plot(fpr2, tpr2,label='Random Forest ROC Curve (area = %.4f)' % auc_rf, 
         	color = 'red', 
         	linestyle=':', 
         	linewidth=2)
	plt.plot(fpr3, tpr3,label='Neural Networks ROC Curve (area = %.4f)' % auc_nn, 
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

<img src="images/rocNotebook.png" style="width: 100px;"/>

Let's zoom in to get a better picture!

### ROC Curves Plot Zoomed in

	fig, ax = plt.subplots(figsize=(10, 10))
	plt.plot(fpr, tpr, label='Kth-NN ROC Curve  (area = %.4f)' % auc_knn, 
         	color = 'deeppink', 
         	linewidth=1)
	plt.plot(fpr1, tpr1,label='Decision Trees ROC Curve  (area = %.4f)' % auc_dt, 
         	color = 'navy', 
         	linewidth=2)
	plt.plot(fpr2, tpr2,label='Random Forest ROC Curve  (area = %.4f)' % auc_rf, 
         	color = 'red', 
         	linestyle=':', 
         	linewidth=3)
	plt.plot(fpr3, tpr3,label='Neural Networks ROC Curve  (area = %.4f)' % auc_nn, 
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

<img src="images/rocClose.png" style="width: 100px;"/>

From the `auc` calculations we can see that both *Random Forest* and *Neural Networks* performed better than *Kth Nearest Neighbor* and *Decision Trees* which is pretty intuitive. 

Also visually examining the plot, *Random Forest* is noticeable more elevated than the other models which is indicative of a good prediction tool, using this form of diagnositcs. I will go into more detail later on, but for now this will do. 

# Conclusions

As you can see for this project, **Kth Nearest Neighbor** performed significantly better with respect to the *test error rate*. But did the worst with respect to **False Negatives** (9 total, that's a lot!). **Kth Nearest Neighbor** also doesn't really give us much insight as to what variables were able to indicate which specimen's were *Malignant* or *Benign*, which is something that might be useful for the people in the field of Cancer research, and/or if this were real world application the people who hired you to do this analysis.  

So when drawing conclusions although a model could perform better than another, often we look to see which can tell us more of our data and ultimately **Random Forest**, was the model that was able to tell us more about the variable interaction while at the same time providing us with the smallest **Test Error Rate** and **False Negative**.

| Model/Algorithm | Test Error Rate | False Negative for Test Set | Area under the Curve for ROC |
|-----------------|-----------------|--------------------------------------------|----------------|
| Kth Nearest Neighbor | 3.509% | 2| 0.9627 | 
| Decision Trees | 5.263% | 4 | 0.9482 | 
| Random Forest | 3.509% | 3 | 0.9673 | 
| Neural Networks | 2.632% | 1 | 0.9701 | 

Fin :)
