import sys
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

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
        my_estimator = my_estimators[estimator]
    except KeyError as e:
        print('Model specified not found in dictionary.\nAvailable options are: {0}'
              .format(list(my_estimators)))
        raise KeyError(estimator)
    try:
        check_is_fitted(fit, my_estimator)
    except (NotFittedError, TypeError) as e:
        print(e)
        raise e
    # Outputting predictions and prediction probability
    # for test set
    predictions = fit.predict(test_set)
    accuracy = fit.score(test_set, test_class_set)
    predictions_prob = fit.predict_proba(test_set)[:, 1]
    # ROC Curve stuff
    fpr, tpr, _ = roc_curve(test_class_set,
            predictions_prob)
    auc_fit = auc(fpr, tpr)
    return {'predictions': predictions,
    'accuracy': accuracy,
    'fpr': fpr,
    'tpr': tpr,
    'auc': auc_fit}
