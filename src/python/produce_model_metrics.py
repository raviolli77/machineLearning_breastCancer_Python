import sys
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

# Function for All Models to produce Metrics ---------------------

def produce_model_metrics(my_fit, test_set, test_class_set, estimator):
    my_estimators = {
    'random_forest': 'estimators_',
    'neural_network': 'out_activation_',
    'kth_nearest_neighor': '_fit_method'
    }
    try:
        check_is_fitted(my_fit, my_estimators[estimator])
    except NotFittedError as e:
        print(e)
        sys.exit(1)
    # Outputting predictions and prediction probability
    # for test set
    predictions = my_fit.predict(test_set)
    accuracy = my_fit.score(test_set, test_class_set)
    predictions_prob = my_fit.predict_proba(test_set)[:, 1]
    # ROC Curve stuff
    fpr, tpr, _ = roc_curve(test_class_set,
            predictions_prob)
    auc_fit = auc(fpr, tpr)
    return {'predictions': predictions,
    'accuracy': accuracy,
    'fpr': fpr,
    'tpr': tpr,
    'auc': auc_fit}
