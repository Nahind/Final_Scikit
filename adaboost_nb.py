from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import SDK as sdk
import personal_settings

path = personal_settings.PATH
algorithm = "adaboost_decision_tree"

extraction_type = "MSD-JMIRMFCCS"
print("Starting new classification. Extraction method : " + extraction_type)
folder = path + extraction_type + "/"

try :
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(), algorithm="SAMME", n_estimators=500)
    sdk.evaluate_classifier(bdt, folder, extraction_type, algorithm)
except Exception as e:
    print(str(e))
    pass

print("Ended extraction : " + extraction_type)
