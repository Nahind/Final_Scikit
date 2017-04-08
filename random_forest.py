import personal_settings
from sklearn import ensemble
import os
import SDK as sdk

path = personal_settings.PATH
algorithm = os.path.basename(__file__).split(".py")[0]
datasets = ["MSD-SSD"]
# datasets = os.listdir(path)

for extraction_type in datasets:
    print("Starting new classification. Extraction method : " + extraction_type)
    folder = path + extraction_type + "/"

    try:
        for n_trees in [100]:
            clf = ensemble.RandomForestClassifier(n_jobs=-1, verbose=20, n_estimators=n_trees)
            sdk.evaluate_classifier(clf, folder, extraction_type, algorithm, suffixe="_"+str(n_trees)+"_trees")

    except Exception as e:
        print(str(e))
        pass
print("Ended extraction : " + extraction_type)