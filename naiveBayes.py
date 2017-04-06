from sklearn.naive_bayes import GaussianNB
import personal_settings
import SDK as sdk
import os

path = personal_settings.PATH
algorithm = "naive_bayes"

# execute for all datasets:
for extraction_type in os.listdir(path):

    print("Starting new classification. Extraction method : " + extraction_type)
    folder = path + extraction_type + "/"

    try:
        clf = GaussianNB()
        sdk.evaluate_classifier(clf, folder, extraction_type, algorithm)
    except Exception as e:
        print(str(e))
        pass

    print("Ended extraction : " + extraction_type)
