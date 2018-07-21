import sys
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Function for All Models to produce Metrics ---------------------

def produce_model_metrics(fit, test_set, test_class_set, estimator):
    """
    Purpose
    ----------
    Function that will return predictions and probability metrics for said
    predictions.

    Parameters
    ----------
    * fit: 	Fitted model containing the attribute feature_importances_
    * test_set: dataframe/array containing the test set values
    * test_class_set: array containing the target values for the test set
    * estimator: String represenation of appropriate model, can only contain the
    following: ['knn', 'rf', 'nn']

    Returns
    ----------
    Box plot graph for all numeric data in data frame
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


    # Outputting predictions and prediction probability
    # for test set
    predictions = fit.predict(test_set)
    accuracy = fit.score(test_set, test_class_set)
    # We grab the second array from the output which corresponds to
    # to the predicted probabilites of positive classes
    # Ordered wrt fit.classes_ in our case [0, 1] where 1 is our positive class
    predictions_prob = fit.predict_proba(test_set)[:, 1]
    # ROC Curve stuff
    fpr, tpr, _ = roc_curve(test_class_set,
            predictions_prob,
            pos_label = 1)
    auc_fit = auc(fpr, tpr)
    return {'predictions': predictions,
    'accuracy': accuracy,
    'fpr': fpr,
    'tpr': tpr,
    'auc': auc_fit}
