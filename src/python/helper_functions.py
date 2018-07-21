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
Helper Functions Script
"""
# Import Packages -----------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_extraction import names_index
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def print_target_perc(data_frame, col):
    """Function used to print class distribution for our data set"""
    try:
        # If the number of unique instances in column exceeds 20 print warning
        if data_frame[col].nunique() > 20:
            return print('Warning: there are {0} values in `{1}` column which exceed the max of 20 for this function. \
                         Please try a column with lower value counts!'
                         .format(data_frame[col].nunique(), col))
        # Stores value counts
        col_vals = data_frame[col].value_counts().sort_values(ascending=False)
        # Resets index to make index a column in data frame
        col_vals = col_vals.reset_index()

        # Create a function to output the percentage
        f = lambda x, y: 100 * (x / sum(y))
        for i in range(0, len(col_vals['index'])):
            print('`{0}` accounts for {1:.2f}% of the {2} column'\
                  .format(col_vals['index'][i],
                          f(
                              col_vals[col].iloc[i],
                              col_vals[col]),
                          col))
    # try-except block goes here if it can't find the column in data frame
    except KeyError as e:
        raise KeyError('{0}: Not found. Please choose the right column name!'.format(e))

def plot_box_plot(data_frame, data_set, xlim=None):
    """
    Purpose
    ----------
    Creates a seaborn boxplot including all dependent
    variables and includes x limit parameters

    Parameters
    ----------
    * data_frame : Name of pandas.dataframe
    * data_set : Name of title for the boxplot
    * xlim : Set upper and lower x-limits

    Returns
    ----------
    Box plot graph for all numeric data in data frame
    """
    f, ax = plt.subplots(figsize=(11, 15))

    ax.set_axis_bgcolor('#fafafa')
    if xlim is not None:
        plt.xlim(*xlim)
    plt.ylabel('Dependent Variables')
    plt.title("Box Plot of {0} Data Set"\
              .format(data_set))
    ax = sns.boxplot(data = data_frame.select_dtypes(include = ['number']),
                     orient = 'h')

    plt.show()
    plt.close()

def normalize_data_frame(data_frame):
    """
    Purpose
    ----------
    Function created to normalize data set.
    Intializes an empty data frame which will normalize all columns that
    have at > 10 unique values (chosen arbitrarily since target columns
    will have classes < 10) and append the non-float types.
    Application can vary significantly for different data set, use with caution
    or modify accordingly.

    Parameters
    ----------
    * data_frame: 	Name of pandas.dataframe

    Returns
    ----------
    * data_frame_norm:	Normalized dataframe values ranging (0, 1)
    """
    data_frame_norm = pd.DataFrame()
    for col in data_frame:
        if ((len(np.unique(data_frame[col])) > 10) & (data_frame[col].dtype != 'object')):
            data_frame_norm[col]=((data_frame[col] - data_frame[col].min()) /
                                  (data_frame[col].max() - data_frame[col].min()))
        else:
            data_frame_norm[col] = data_frame[col]
    return data_frame_norm



def variable_importance(fit):
    """
    Purpose
    ----------
    Checks if model is fitted CART model then produces variable importance
    and respective indices in dictionary.

    Parameters
    ----------
    * fit:  Fitted model containing the attribute feature_importances_

    Returns
    ----------
    Dictionary containing arrays with importance score and index of columns
    ordered in descending order of importance.
    """
    try:
        if not hasattr(fit, 'fit'):
            return print("'{0}' is not an instantiated model from scikit-learn".format(fit))

        # Captures whether the model has been trained
        if not vars(fit)["estimators_"]:
            return print("Model does not appear to be trained.")
    except KeyError:
        KeyError("Model entered does not contain 'estimators_' attribute.")

    importances = fit.feature_importances_
    indices = np.argsort(importances)[::-1]
    return {'importance': importances,
            'index': indices}

def print_var_importance(importance, indices, name_index):
    """
    Purpose
    ----------
    Prints dependent variable names ordered from largest to smallest
    based on information gain for CART model.
    Parameters
    ----------
    * importance: Array returned from feature_importances_ for CART
                    models organized by dataframe index
    * indices: Organized index of dataframe from largest to smallest
                    based on feature_importances_
    * name_index: Name of columns included in model

    Returns
    ----------
    Prints feature importance in descending order
    """
    print("Feature ranking:")

    for f in range(0, indices.shape[0]):
        i = f
        print("{0}. The feature '{1}' has a Mean Decrease in Impurity of {2:.5f}"
              .format(f + 1,
                      names_index[indices[i]],
                      importance[indices[f]]))

def variable_importance(fit):
    """
    Purpose
    ----------
    Checks if model is fitted CART model then produces variable importance
    and respective indices in dictionary.
    Parameters
    ----------
    * fit:  Fitted model containing the attribute feature_importances_
    Returns
    ----------
    Dictionary containing arrays with importance score and index of columns
    ordered in descending order of importance.
    """
    try:
        if not hasattr(fit, 'fit'):
            return print("'{0}' is not an instantiated model from scikit-learn".format(fit))

        # Captures whether the model has been trained
        if not vars(fit)["estimators_"].all():
            return print("Model does not appear to be trained.")
    except KeyError:
        raise KeyError("Model entered does not contain 'estimators_' attribute.")

    importances = fit.feature_importances_
    indices = np.argsort(importances)[::-1]
    return {'importance': importances,
            'index': indices}

def print_var_importance(importance, indices, name_index):
    """
    Purpose
    ----------
    Prints dependent variable names ordered from largest to smallest
    based on information gain for CART model.
    Parameters
    ----------
    * importance: Array returned from feature_importances_ for CART
                    models organized by dataframe index
    * indices: Organized index of dataframe from largest to smallest
                    based on feature_importances_
    * name_index: Name of columns included in model
    Returns
    ----------
    Prints feature importance in descending order
    """
    print("Feature ranking:")

    for f in range(0, indices.shape[0]):
        i = f
        print("{0}. The feature '{1}' has a Mean Decrease in Impurity of {2:.5f}"
              .format(f + 1,
                      name_index[indices[i]],
                      importance[indices[f]]))

def variable_importance_plot(importance, indices, name_index):
    """
    Purpose
    ----------
    Prints bar chart detailing variable importance for CART model
    NOTE: feature_space list was created because the bar chart
    was transposed and index would be in incorrect order.
    Parameters
    ----------
    * importance: Array returned from feature_importances_ for CART
                    models organized by dataframe index
    * indices: 	Organized index of dataframe from largest to smallest
                    based on feature_importances_
    * name_index: Name of columns included in model

    Returns:
    ----------
    Returns variable importance plot in descending order
    """
    index = np.arange(len(name_index))

    importance_desc = sorted(importance)
    feature_space = []
    for i in range(indices.shape[0] - 1, -1, -1):

    feature_space.append(name_index[indices[i]])

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_facecolor('#fafafa')
    plt.title('Feature importances for Gradient Boosting Model\
    \nCustomer Churn')
    plt.barh(index,
             importance_desc,
             align="center",
             color = '#875FDB')
    plt.yticks(index,
               feature_space)

    plt.ylim(-1, indices.shape[0])
    plt.xlim(0, max(importance_desc) + 0.01)
    plt.xlabel('Mean Decrease in Impurity')
    plt.ylabel('Feature')

    plt.show()
    plt.close()


def plot_roc_curve(fpr, tpr, auc, estimator, xlim=None, ylim=None):
    """
    Purpose
    ----------
    Function creates ROC Curve for respective model given selected parameters.
    Optional x and y limits to zoom into graph

    Parameters
    ----------
    * fpr: Array returned from sklearn.metrics.roc_curve for increasing
    false positive rates
    * tpr: Array returned from sklearn.metrics.roc_curve for increasing
    true positive rates
    * auc: Float returned from sklearn.metrics.auc (Area under Curve)
    * estimator: String represenation of appropriate model, can only contain the
    following: ['knn', 'rf', 'nn']
    * xlim: Set upper and lower x-limits
    * ylim: Set upper and lower y-limits
    """
    my_estimators = {'knn': ['Kth Nearest Neighbor', 'deeppink'],
              'rf': ['Random Forest', 'red'],
              'nn': ['Neural Network', 'purple']}

    try:
        plot_title = my_estimators[estimator][0]
        color_value = my_estimators[estimator][1]
    except KeyError as e:
        raise("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \
              Please refer to function to check `my_estimators` dictionary.".format(estimator))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_bgcolor('#fafafa')

    plt.plot(fpr, tpr,
             color=color_value,
             linewidth=1)
    plt.title('ROC Curve For {0} (AUC = {1: 0.3f})'\
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

def cross_val_metrics(fit, training_set, class_set, estimator, print_results = True):
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

    Returns
    ----------
    scores.mean(): Float representing cross validation score
    scores.std() / 2: Float representing the standard error (derived
                from cross validation score's standard deviation)
    """
    my_estimators = {
    'rf': 'estimators_',
    'nn': 'out_activation_',
    'knn': '_fit_method'
    }
    try:
        # Captures whether first parameter is a model
        if not hasattr(fit, 'fit'):
            return print("'{0}' is not an instantiated model from scikit-learn".format(fit))

        # Captures whether the model has been trained
        if not vars(fit)[my_estimators[estimator]]:
            return print("Model does not appear to be trained.")

    except KeyError as e:
        raise KeyError("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \
              Please refer to function to check `my_estimators` dictionary.".format(estimator))

    n = KFold(n_splits=10)
    scores = cross_val_score(fit,
                         training_set,
                         class_set,
                         cv = n)
    if print_results:
        for i in range(0, len(scores)):
            print("Cross validation run {0}: {1: 0.3f}".format(i, scores[i]))
        print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
              .format(scores.mean(), scores.std() / 2))
    else:
        return scores.mean(), scores.std() / 2


def create_conf_mat(test_class_set, predictions):
    """Function returns confusion matrix comparing two arrays"""
    if (len(test_class_set.shape) != len(predictions.shape) == 1):
        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')
    elif (test_class_set.shape != predictions.shape):
        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')
    else:
        # Set Metrics
        test_crosstb_comp = pd.crosstab(index = test_class_set,
                                        columns = predictions)
        test_crosstb = test_crosstb_comp.values
        return test_crosstb
