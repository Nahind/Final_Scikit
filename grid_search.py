"""
============================================================
Parameter estimation using grid search with cross-validation
============================================================

This examples shows how a classifier is optimized by cross-validation,
which is done using the :class:`sklearn.model_selection.GridSearchCV` object
on a development set that comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained model is
then measured on a dedicated evaluation set that was not used during
the model selection step.

More details on tools available for model selection can be found in the
sections on :ref:`cross_validation` and :ref:`grid_search`.

"""

from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import SDK as sdk
import personal_settings

path = personal_settings.PATH
extraction_type = "MSD-JMIRMFCCS"
folder = path + extraction_type + "/"
algorithm = "grid_search_svc"

# Loading the Digits dataset
#  digits = datasets.load_digits()
training, valid = sdk.get_test_dataset(path)

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
# n_samples = len(digits.images)
# X = digits.images.reshape((n_samples, -1))
# y = digits.target
X = valid.data
y = valid.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10]},
                    {'kernel': ['linear'], 'C': [1, 10]}]

scores = ['precision', 'recall']
# file to save all scoring data
file = open(algorithm + "_" + extraction_type.lower() + ".txt", "w")

for score in scores:
    file.write("# Tuning hyper-parameters for %s" % score)
    print("# Tuning hyper-parameters for %s" % score)
    file.write("")
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    file.write("Best parameters set found on development set:")
    print("Best parameters set found on development set:")
    file.write("")
    print()
    file.write(clf.best_params_)
    print(clf.best_params_)
    file.write("")
    print()
    file.write("Grid scores on development set:")
    print("Grid scores on development set:")
    file.write("")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        file.write("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        file.write("")
        print()

        file.write("Detailed classification report:")
        print("Detailed classification report:")
        file.write("")
        print()
        file.write("The model is trained on the full development set.")
        print("The model is trained on the full development set.")
        file.write("The scores are computed on the full evaluation set.")
        print("The scores are computed on the full evaluation set.")
        file.write("")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        file.write(classification_report(y_true, y_pred))
        print(classification_report(y_true, y_pred))
        file.write("")
        print()

        # Note the problem is too easy: the hyperparameter plateau is too flat and the
        # output model is the same for precision and recall with ties in quality.
