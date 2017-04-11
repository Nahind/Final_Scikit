from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import SDK as sdk
import numpy as np
import os
import personal_settings

path = personal_settings.PATH
algorithm = os.path.basename(__file__).split(".py")[0]
datasets = personal_settings.LARGE_DATASETS


#execute for all datasets:

# for extraction_type in os.listdir(path):
for extraction_type in datasets:

    print("Starting new classification. Extraction method : " + extraction_type)
    folder = path + extraction_type + "/"

    try:
        training, validation = sdk.load_dataset_from_folder(folder, extraction_type)
        scaler = StandardScaler()
        # Normalize training & validation sets
        print("Start normalizing data")
        scaler.fit(training.data)
        training.data = scaler.transform(training.data)
        print("Training data has been normalized")
        validation.data = scaler.transform(validation.data)
        print("Validation data has been normalized")

        X = training.data
        y = training.target

        # Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=20)

        C_range = 10.0 ** np.arange(-4, 6)
        gamma_range = 10.0 ** np.arange(-4, 3)
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range}]

        # file to save all scoring data
        file = open(algorithm + "_" + extraction_type.lower() + ".txt", "w")

        print("# Tuning hyper-parameters for accuracy")
        file.write("# Tuning hyper-parameters for accuracy" + "\n")
        print()

        clf = GridSearchCV(SVC(C=1, verbose=20), tuned_parameters, cv=10, verbose=20, n_jobs=-1, scoring='accuracy')
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
            file.write("%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params))

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
    print("Ended extraction : " + extraction_type)

