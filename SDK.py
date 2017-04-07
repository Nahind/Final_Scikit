from scipy.io import arff
from sklearn import metrics
from sklearn.externals import joblib
import numpy as np
import os

class Data:

    def __init__(self, data, meta):
        d = data[meta.names()[:-1]]
        self.data = d.view(np.float).reshape(d.shape + (-1,))
        self.target = data['class']

def save_model(model, extraction_type, algorithm):
    dir = "./output_" + algorithm + "/models/"
    if not os.path.exists(dir):
        print("Creating new directory : " + dir)
        os.makedirs(dir)
    print("Start model saving")
    joblib.dump(model, dir + extraction_type.lower() +'.pkl')
    print("Model has been saved")


def save_model(model, extraction_type, algorithm):
    dir = "./output_" + algorithm + "/models/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    joblib.dump(model, dir + extraction_type.lower() +'.pkl')


def save_classification_report(validation, extraction_type, y_pred, algorithm, **kwargs):
    dir = "./output_" + algorithm + "/evaluations/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    if ('suffixe' in kwargs):
        file = open(dir + extraction_type.lower() + kwargs['suffixe'] + ".txt", "w")
    else:
        file = open(dir + extraction_type.lower() + ".txt", "w")
    accuracy = metrics.accuracy_score(validation.target, y_pred)
    file.write("Classifications correctes : " + accuracy.__str__() + "%\n")
    file.write("Classifications incorrectes : " + (1 - accuracy).__str__() + "%\n")
    file.writelines(metrics.classification_report(validation.target, y_pred))
    file.close()


def loadData(fileName):
    dataFolder = "/home/nandane/Documents/Cours_ETS_MTL/LOG770_Intelligence_machine/LAB4/DEV_PREPARED/MSD-JMIRMOMENTS/"
    trainingFile = dataFolder + "msd-jmirmoments_dev_train.arff"
    evaluationFile = dataFolder + "msd-jmirmoments_dev_val.arff"

    trainData, trainMeta = arff.loadarff(trainingFile)
    training = Data(trainData, trainMeta)

    valData, valMeta = arff.loadarff(evaluationFile)
    validation = Data(valData, valMeta)

    return training, validation


def load_data_from_folder(path, extraction_type):
    prefix = extraction_type.lower()
    training_file = path + prefix + "_dev_train.arff"
    evaluationFile = path + prefix + "_dev_val.arff"

    trainData, trainMeta = arff.loadarff(training_file)
    training = Data(trainData, trainMeta)

    valData, valMeta = arff.loadarff(evaluationFile)
    validation = Data(valData, valMeta)

    return training, validation


def evaluate_classifier(clf, folder, extraction_type, algorithm, **kwargs):
    training, validation = load_data_from_folder(folder, extraction_type)
    model = clf.fit(training.data, training.target)
    print("Training has ended")
    save_model(model, extraction_type, algorithm)
    y_pred = model.predict(validation.data)
    save_classification_report(validation, extraction_type, y_pred, algorithm, **kwargs)


def test_classifier_params(clf, folder, extraction_type, algorithm):
    training, validation = load_data_from_folder(folder, extraction_type)
    model = clf.fit(training.data, training.target)
    save_model(model, extraction_type, algorithm)
    y_pred = model.predict(validation.data)
    save_classification_report(validation, extraction_type, y_pred, algorithm)



