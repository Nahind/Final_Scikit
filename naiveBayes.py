from sklearn.naive_bayes import GaussianNB
import SDK as sdk
import os

path = "/home/nandane/Documents/Cours_ETS_MTL/LOG770_Intelligence_machine/LAB4/DEV_PREPARED/"
algorithm = "naive_bayes"

# execute for all datasets:
for extraction_type in os.listdir(path):

    print("Starting new classification. Extraction method : " + extraction_type)
    folder = path + extraction_type + "/"

    try :
        clf = GaussianNB()
        sdk.evaluate_classifier(clf, folder, extraction_type, algorithm)
    except FileNotFoundError:
        print("File of extraction type : " + extraction_type + " not found !")

    print("Ended extraction : " + extraction_type)
