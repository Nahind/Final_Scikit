###
# Build of a framework base on the approach of a high kaggle competitor
# http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/
###

from scipy.io import arff
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB

import SDK as sdk

import numpy as np
import personal_settings
import os


path = personal_settings.PATH
extraction_type = "MSD-JMIRMFCCS"
folder = path + extraction_type + "/"
algorithm = "ml_framework"

class Data:
    def __init__(self, data, meta):
        d = data[meta.names()[:-1]]
        self.data = d.view(np.float).reshape(d.shape + (-1,))
        self.target = data['class']


def load_dataset_from_folder(folder):
    extraction_type = folder.split("/")[len(folder.split("/"))-2]
    prefix = extraction_type.lower()
    training_file = folder + prefix + "_dev_train.arff"
    evaluation_file = folder + prefix + "_dev_val.arff"
    print("Start loading training data")
    train_data, train_meta = arff.loadarff(training_file)
    training = Data(train_data, train_meta)
    print("Start loading validation data")
    val_data, val_meta = arff.loadarff(evaluation_file)
    validation = Data(val_data, val_meta)
    print("Data has been loaded successfully")
    return training, validation


def load_test_file(folder, extraction_type):
    prefix = extraction_type.lower()
    test_file = folder + prefix + "_test_nolabels.arff"
    print("Start loading training data")
    test_data, test_meta = arff.loadarff(test_file)
    d = test_data[test_meta.names()[:-1]]
    testing = d.view(np.float).reshape(d.shape + (-1,))

    return testing


def save_classification_report(y_valid, extraction_type, y_pred, algorithm, **kwargs):
    file = sdk.get_or_create_output_file(extraction_type, algorithm, **kwargs)
    print("Start computing information")
    accuracy = metrics.accuracy_score(y_valid, y_pred)
    file.write("Classifications correctes : " + accuracy.__str__() + "%\n")
    file.write("Classifications incorrectes : " + (1 - accuracy).__str__() + "%\n")
    file.writelines(metrics.classification_report(y_valid, y_pred))
    file.close()
    print("Classification report has been saved !")



def select_best_features_with_trees(X, y, dataset_name, overwrite=False):
    dataset_name = dataset_name.lower()
    directory = "./output_features_selection"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("directory created : " + directory)

    # if the selection exists, we return it
    for f_name in os.listdir(directory):
        if f_name.startswith(dataset_name) and overwrite is False:
            print("Data has been saved before. We are retrieving it from file : " + dataset_name)
            save_model = joblib.load(directory + "/" + dataset_name + '.pkl')
            return save_model.transform(X)

    # features selection
    clf = RandomForestClassifier(n_estimators=5, n_jobs=-1)
    clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    X_selected = model.transform(X)
    print("Saving data for future use in : " + directory)
    joblib.dump(model, directory + "/" + dataset_name + '.pkl')
    print("We went from : " + str(X.shape[1]) + " features to " + str(X_selected.shape[1]))

    return X_selected



def merge_datasets_best_features(path, datasets):

    X_train_merged = []
    X_valid_merged = []

    for dataset in datasets:
        print("dealing with : " + dataset)
        folder = path + dataset.upper() + "/"
        training, validation = load_dataset_from_folder(folder)
        X_train, y_train = training.data, training.target
        X_valid, y_valid = validation.data, validation.target
        X_train = select_best_features_with_trees(X_train, y_train, dataset)
        X_valid = select_best_features_with_trees(X_valid, y_valid, dataset)
        X_train_merged.append(X_train)
        X_valid_merged.append(X_valid)
        print("merged train shape = " + str(np.concatenate(X_train_merged, axis=1).shape))
        print("merged valid shape = " + str(np.concatenate(X_valid_merged, axis=1).shape))

    return np.concatenate(X_train_merged, axis=1), y_train, np.concatenate(X_valid_merged, axis=1), y_valid


def merge_datasets_best_features_test(path, datasets):

    test_merged = []

    for dataset in datasets:
        print("dealing with : " + dataset)
        folder = path + dataset.upper() + "/"
        testing = load_test_file(folder, dataset.upper())
        testing = select_best_features_with_trees(testing, None, dataset)
        test_merged.append(testing)

        print("merged train shape = " + str(np.concatenate(test_merged, axis=1).shape))

    return np.concatenate(test_merged, axis=1)

###
# Testing script
###

folder_test = "/home/nandane/Documents/Cours_ETS_MTL/LOG770_Intelligence_machine/LAB4/DEV_PREPARED_TEST/"
merge_d = personal_settings.BEST_DATASETS
X_train, y_train, X_valid, y_valid = merge_datasets_best_features(path, merge_d)
X_test = merge_datasets_best_features_test(folder_test, merge_d)

try:
    n_trees = 550
    clf_rd = RandomForestClassifier(n_estimators=n_trees, n_jobs=5, verbose=20)
    print("Start training")
    model = clf_rd.fit(X_train, y_train)
    print("Start predicting validation set")
    y_pred = model.predict(X_test)
    f = open("./results.txt", 'w')
    for i in y_pred:
        f.write(str(i).replace("\'", "").replace("b", "") + "\n")


except Exception as e:
    print(str(e))
    pass