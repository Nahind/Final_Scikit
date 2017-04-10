from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn.ensemble import VotingClassifier

import SDK as sdk
import os
import personal_settings

path = personal_settings.PATH
algorithm = os.path.basename(__file__).split(".py")[0]
datasets = os.listdir(path)
datasets = ["MSD-SSD"]

#execute for all datasets:
for extraction_type in datasets:
    print("Starting new classification. Extraction method : " + extraction_type)
    folder = path + extraction_type + "/"

    try:

        # Training classifiers
        clf1 = MLPClassifier(verbose=True, hidden_layer_sizes=300, early_stopping=True)
        clf2 = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
        clf3 = ensemble.RandomForestClassifier(n_jobs=-1, verbose=20, n_estimators=50)
        eclf = VotingClassifier(estimators=[('mlp', clf1), ('knn', clf2), ('rf', clf3)], voting='soft',
                                weights=[2, 1, 2], n_jobs=-1)
        sdk.evaluate_classifier(eclf, folder, extraction_type, algorithm)


    except Exception as e:
        print(str(e))
        pass
    print("Ended extraction : " + extraction_type)
