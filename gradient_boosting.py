from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import personal_settings
import SDK as sdk
import os

path = personal_settings.PATH
algorithm = os.path.basename(__file__).split(".py")[0]

# execute for all datasets:
for extraction_type in personal_settings.BEST_DATASETS:

    print("Starting new classification. Extraction method : " + extraction_type)
    folder = path + extraction_type + "/"

    try:
        clf = GaussianNB()
        gbclf = GradientBoostingClassifier(verbose=20)
        sdk.evaluate_classifier(gbclf, folder, extraction_type, algorithm)
    except Exception as e:
        print(str(e))
        pass

    print("Ended extraction : " + extraction_type)
