from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB

import SDK as sdk
import os
import personal_settings

path = personal_settings.PATH
algorithm = os.path.basename(__file__).split(".py")[0]
datasets = os.listdir(path)
datasets = ["MSD-SSD"]

pca_training_sets = [0] * 4
pca_validation_sets = [0] * 4
counter = -1

for extraction_type in personal_settings.SMALL_DATASETS[:4]:
    print("Starting new classification. Extraction method : " + extraction_type)
    folder = path + extraction_type + "/"
    clf = GaussianNB()
    counter += 1


    try:

        training, validation = sdk.load_dataset_from_folder(path, datasets)
        print("Start training MLP Classifier")
        pca = PCA(2)
        pca_t = pca.fit_transform(training.data)
        pca_v = pca.fit_transform(validation.data)
        pca_training_sets[counter] = pca_t
        pca_validation_sets[counter] = pca_v

        # print("Training has ended")
        # # Save MLP model
        # sdk.save_model(model, extraction_type, algorithm)
        # # Evaluation model
        # print("Start predicting validation set")
        # y_pred = model.predict(validation.data)
        # # Save Evaluation report
        # sdk.save_classification_report(validation, extraction_type, y_pred, algorithm, suffixe="_no_early_stop")
        #

    except Exception as e:
        print(str(e))
        pass
    print("Ended extraction : " + extraction_type)
