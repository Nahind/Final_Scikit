from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import SDK as sdk
import personal_settings

path = personal_settings.PATH
extraction_type = "MSD-JMIRMOMENTS"
folder = path + extraction_type + "/"
algorithm = "grid_search_svc"

for extraction_type in personal_settings.LARGE_DATASETS:

    print("Starting new classification. Extraction method : " + extraction_type)
    folder = path + extraction_type + "/"
    training, valid = sdk.load_dataset_from_folder(folder, extraction_type)

    X = valid.data
    y = valid.target

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=5)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]}]

    scores = ['accuracy']

    # file to save all scoring data
    file = open(algorithm + "_" + extraction_type.lower() + ".txt", "w")

    for score in scores:

        try:
            print("# Tuning hyper-parameters for %s" % score)
            file.write("# Tuning hyper-parameters for %s" % score + "\n")
            print()

            clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, verbose=20, n_jobs=-1, scoring=score)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            file.write("Best parameters set found on development set:" + "\n")
            print()
            print(clf.best_params_)
            file.write("best params = " + str(clf.best_params_) + "\n")
            print()
            print("Grid scores on development set:")
            file.write("Grid scores on development set:" + "\n")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']

            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

            print()
            print("Detailed classification report:")
            file.write("Detailed classification report:" + "\n")
            print()
            print("The model is trained on the full development set.")
            file.write("The model is trained on the full development set." + "\n")
            print("The scores are computed on the full evaluation set.")
            file.write("The scores are computed on the full evaluation set." + "\n")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            accuracy = accuracy_score(y_true, y_pred)
            print("accuracy = " + accuracy.__str__())
            file.write("accuracy = " + accuracy.__str__() + "\n")
            print()
            print(classification_report(y_true, y_pred))
            file.writelines(classification_report(y_true, y_pred) + "\n")
            print()

        except Exception as e:
            print(str(e))
            pass

