#!/usr/bin/env python3

#####################################################
##    WISCONSIN BREAST CANCER MACHINE LEARNING     ##
#####################################################
#
# Project by Raul Eulogio
#
# Project found at: https://www.inertia7.com/projects/3
#

"""
Exploratory Analysis
"""
import helper_functions as hf
import matplotlib.pyplot as plt 
import seaborn as sns 

breast_cancer = hf.breast_cancer

print('''
	########################################
	##    DATA FRAME SHAPE AND DTYPES     ##
	########################################
	''')

print("Here's the dimensions of our data frame:\n", 
	breast_cancer.shape)

print("Here's the data types of our columns:\n", 
	breast_cancer.dtypes)	

print("Some more statistics for our data frame: \n", 
	breast_cancer.describe())

print('''
##########################################
##      STATISTICS RELATING TO DX       ##
##########################################
''')

# Let's look at the count of the new representations of our Dx's
print("Count of the Dx:\n", breast_cancer['diagnosis']\
	.value_counts())

# Next let's use the helper function to show distribution
# of our data frame
hf.calc_diag_percent(breast_cancer, 'diagnosis')

# Scatterplot Matrix
# Variables chosen from Random Forest modeling.

cols = ['concave_points_worst', 'concavity_mean', 
	'perimeter_worst', 'radius_worst', 
	'area_worst', 'diagnosis']

sns.pairplot(breast_cancer,
	x_vars = cols,
	y_vars = cols,
	hue = 'diagnosis', 
	palette = ('Red', '#875FDB'), 
	markers=["o", "D"])

plt.title('Scatterplot Matrix')
plt.show()
plt.close()

# Pearson Correlation Matrix
corr = breast_cancer.corr(method = 'pearson') # Correlation Matrix	
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 275, as_cmap=True)	

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,
	cmap=cmap,
	square=True, 
	xticklabels=True, 
	yticklabels=True,
	linewidths=.5, 
	cbar_kws={"shrink": .5}, 
	ax=ax)

plt.title("Pearson Correlation Matrix")
plt.yticks(rotation = 0)
plt.xticks(rotation = 270)	
plt.show()
plt.close()

# BoxPlot
hf.plot_box_plot(breast_cancer, 'Pre-Processed', (-.05, 50))

# Normalizing data 
breast_cancer_norm = hf.normalize_data_frame(breast_cancer)

# Visuals relating to normalized data to show significant difference
print('''
#################################
## Transformed Data Statistics ##
#################################
''')

print(breast_cancer_norm.describe())

hf.plot_box_plot(breast_cancer_norm, 'Transformed', (-.05, 1.05))