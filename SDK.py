from scipy.io import arff
from sklearn import metrics
from sklearn.externals import joblib
import numpy as np
import os


# class representing a labeled data set
# data being the values & target the labels
class Data:
    def __init__(self, data, meta):
        d = data[meta.names()[:-1]]
        self.data = d.view(np.float).reshape(d.shape + (-1,))
        self.target = data['class']


def get_or_create_output_file(extraction_type, algorithm, **kwargs):
    directory = "./output_" + algorithm + "/evaluations/"
    print("Start saving classification report")
    if not os.path.exists(directory):
        print("directory created : " + directory)
        os.makedirs(directory)
    if 'suffixe' in kwargs:
        file = open(directory + extraction_type.lower() + kwargs['suffixe'] + ".txt", "w")
    else:
        file = open(directory + extraction_type.lower() + ".txt", "w")
    return file


def save_model(model, extraction_type, algorithm):
    directory = "./output_" + algorithm + "/models/"
    if not os.path.exists(directory):
        print("Creating new directory : " + directory)
        os.makedirs(directory)
    print("Start model saving")
    joblib.dump(model, directory + extraction_type.lower() + '.pkl')
    print("Model has been saved")


def save_classification_report(validation, extraction_type, y_pred, algorithm, **kwargs):
    file = get_or_create_output_file(extraction_type, algorithm, **kwargs)
    print("Start computing information")
    accuracy = metrics.accuracy_score(validation.target, y_pred)
    file.write("Classifications correctes : " + accuracy.__str__() + "%\n")
    file.write("Classifications incorrectes : " + (1 - accuracy).__str__() + "%\n")
    file.writelines(metrics.classification_report(validation.target, y_pred))
    file.close()
    print("Classification report has been saved !")


def load_model(extraction_type, algorithm, directory=None):
    if directory is None:
        directory = "./output_" + algorithm + "/models/"
    clf = joblib.load(directory + extraction_type.lower() + '.pkl')
    return clf


def load_dataset_from_folder(path, extraction_type):
    prefix = extraction_type.lower()
    training_file = path + prefix + "_dev_train.arff"
    evaluation_file = path + prefix + "_dev_val.arff"
    print("Start loading training data")
    train_data, train_meta = arff.loadarff(training_file)
    training = Data(train_data, train_meta)
    print("Start loading validation data")
    val_data, val_meta = arff.loadarff(evaluation_file)
    validation = Data(val_data, val_meta)
    print("Data has been loaded successfully")
    return training, validation


def evaluate_classifier(clf, folder, extraction_type, algorithm, **kwargs):
    print("Start evaluating classifier")
    training, validation = load_dataset_from_folder(folder, extraction_type)
    print("Start training model")
    model = clf.fit(training.data, training.target)
    print("Training has ended")
    save_model(model, extraction_type, algorithm)
    print("Start predictions on the validation set")
    y_pred = model.predict(validation.data)
    print("Predictions have ended")
    save_classification_report(validation, extraction_type, y_pred, algorithm, **kwargs)


def get_test_dataset(path):
    extraction_type = "MSD-JMIRMFCCS"
    folder = path + extraction_type + "/"
    training, validation = load_dataset_from_folder(folder, extraction_type)
    return training, validation, extraction_type


def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems

    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        # y_ = np.empty(use_elems)
        # y_.fill(ci)
        y_ = np.array([ci] * use_elems)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys
