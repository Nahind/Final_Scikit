from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import SDK as sdk
import os

path = "/home/nandane/Documents/Cours_ETS_MTL/LOG770_Intelligence_machine/LAB4/DEV_PREPARED/"
algorithm = "adaboost_decision_tree"

extraction_type = "MSD-SSD"
print("Starting new classification. Extraction method : " + extraction_type)
folder = path + extraction_type + "/"

try :
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(), algorithm="SAMME", n_estimators=500)
    sdk.evaluate_classifier(bdt, folder, extraction_type, algorithm)
except FileNotFoundError:
    print("File of extraction type : " + extraction_type + " not found !")

print("Ended extraction : " + extraction_type)