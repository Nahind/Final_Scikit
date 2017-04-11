from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import ensemble
import personal_settings
import SDK as sdk
import os

path = personal_settings.PATH
algorithm = os.path.basename(__file__).split(".py")[0]
datasets = os.listdir(path)
datasets = personal_settings.LARGE_DATASETS

# execute for all datasets:
for extraction_type in datasets:
    print("Starting new classification. Extraction method : " + extraction_type)
    folder = path + extraction_type + "/"
    n_trees = 100

    try:
        in_clf_mlp = MLPClassifier(verbose=True, hidden_layer_sizes=300, early_stopping=True)
        in_clf_rd = ensemble.RandomForestClassifier(n_jobs=-1, verbose=20, n_estimators=n_trees)
        clf = OneVsRestClassifier(in_clf_mlp, n_jobs=-1)
        sdk.evaluate_classifier(clf, folder, extraction_type, algorithm, suffixe="_mlp")

    except Exception as e:
        print(str(e))
        pass

    print("Ended extraction : " + extraction_type)
