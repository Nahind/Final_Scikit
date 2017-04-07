from sklearn import tree
import SDK as sdk
import os
import personal_settings

path = personal_settings.PATH
algorithm = os.path.basename(__file__).split(".py")[0]


# execute for all datasets:
for extraction_type in os.listdir(path):

    print("Starting new classification. Extraction method : " + extraction_type)

    folder = path + extraction_type + "/"

    try:
        clf = tree.DecisionTreeClassifier()
        sdk.evaluate_classifier(clf, folder, extraction_type, algorithm)
    except Exception as e:
        print(str(e))
        pass

    print("Ended extraction : " + extraction_type)
