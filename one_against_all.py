from sklearn.multiclass import OneVsRestClassifier
from sklearn import ensemble
import personal_settings
import SDK as sdk
import os

path = personal_settings.PATH
algorithm = os.path.basename(__file__).split(".py")[0]

# execute for all datasets:
for extraction_type in os.listdir(path):
    print("Starting new classification. Extraction method : " + extraction_type)
    folder = path + extraction_type + "/"

    try:
        clf = OneVsRestClassifier(ensemble.RandomForestClassifier(n_jobs=-1, verbose=20, n_estimators=60), n_jobs=-1)
        sdk.evaluate_classifier(clf, folder, extraction_type, algorithm, suffixe="_rd_60")

    except Exception as e:
        print(str(e))
        pass

    print("Ended extraction : " + extraction_type)
