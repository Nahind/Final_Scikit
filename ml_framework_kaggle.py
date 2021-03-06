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
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
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



###
# Testing script
###

from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

merge_d = personal_settings.BEST_DATASETS
# merge_d = [merge_d[0], merge_d[2]]
# X_train, y_train, X_valid, y_valid = merge_datasets_best_features(path, merge_d)

try:
    clf = GaussianNB()
    hidden_layer_size = 200
    n_trees = 550
    clf_mlp = MLPClassifier(verbose=True, hidden_layer_sizes=hidden_layer_size, early_stopping=False)
    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=15, n_jobs=-1, weights='distance')
    clf_rd = RandomForestClassifier(n_estimators=n_trees, n_jobs=5, verbose=20)
    clf_gd = GradientBoostingClassifier(verbose=20, n_estimators=400)
    # model = clf_rd.fit(X_train, y_train)
    # print("Start predicting validation set")
    # y_pred = model.predict(X_valid)
    # # Save Evaluation report
    # save_classification_report(y_valid, "_".join(merge_d), y_pred, algorithm, suffixe="_gd_400")
    # y_pred = None
    # model = None

    #model = clf_rd.fit(X_train, y_train)
    #sdk.save_model(model,  "_".join(merge_d) + "_rd_550", algorithm)
    print("Start predicting validation set")
    model = sdk.load_model("_".join(merge_d) + "_rd_550", algorithm)
    print("Model loaded")
    y_pred = model.predict(X_valid)
    # Save Evaluation report
    save_classification_report(y_valid, "_".join(merge_d), y_pred, algorithm, suffixe="_rd_550")

except Exception as e:
    print(str(e))
    pass



# training, validation = load_dataset_from_folder(folder)
# X_train, y_train = training.data, training.target
# X_valid, y_valid = validation.data, validation.target
# X_train = select_best_features_with_trees(X_train, y_train, extraction_type, True)
# X_valid = select_best_features_with_trees(X_valid, y_valid, extraction_type, False)
#
#
# print("Starting new classification. Extraction method : " + extraction_type)
#
# try:
#     clf = GaussianNB()
#     model = clf.fit(X_train, y_train)
#     sdk.save_model(model, extraction_type + "_nb", "ml_framework")
#     print("Start predicting validation set")
#     y_pred = model.predict(X_valid)
#     # Save Evaluation report
#     sdk.save_classification_report(validation, extraction_type, y_pred, algorithm,
#                                    suffixe="_nb")
#
# except Exception as e:
#     print(str(e))
#     pass
#
# print("Ended extraction : " + extraction_type)
