###
# Build of a framework base on the approach of a high kaggle competitor
# http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/
###

from scipy.io import arff
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


def select_best_features_with_trees(X, y, dataset_name, overwrite=False):
    dataset_name = dataset_name.lower()
    directory = "./output_features_selection"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("directory created : " + directory)

    # if the selection exists, we return it
    for f_name in os.listdir(directory):
        if f_name.startswith(dataset_name) and overwrite is False:
            print("Data has been saved before. We are retrieving it from file")
            save_model = joblib.load(directory + "/" + dataset_name + '.pkl')
            return save_model.transform(X)

    # features selection
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    X_selected = model.transform(X)
    print("Saving data for future use in : " + directory)
    joblib.dump(model, directory + "/" + dataset_name + '.pkl')
    print("We went from : " + str(X.shape[1]) + " features to " + str(X_selected.shape[1]))

    return X_selected


training, validation = load_dataset_from_folder(folder)
X_train, y_train = training.data, training.target
X_valid, y_valid = validation.data, validation.target
X_train = select_best_features_with_trees(X_train, y_train, extraction_type, True)
X_valid = select_best_features_with_trees(X_valid, y_valid, extraction_type, False)


print("Starting new classification. Extraction method : " + extraction_type)

try:
    clf = GaussianNB()
    model = clf.fit(X_train, y_train)
    sdk.save_model(model, extraction_type + "_nb", "ml_framework")
    print("Start predicting validation set")
    y_pred = model.predict(X_valid)
    # Save Evaluation report
    sdk.save_classification_report(validation, extraction_type, y_pred, algorithm,
                                   suffixe="_nb")

except Exception as e:
    print(str(e))
    pass

print("Ended extraction : " + extraction_type)
