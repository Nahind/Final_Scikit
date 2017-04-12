from sklearn import neighbors
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

    for weights in ['distance']:
        
        for n_neighbors in [3,5,6,7]:

            print("Starting new classification. Extraction method : " + extraction_type)
            print("weights = " + weights)
            
            try:
                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1, weights=weights)
                sdk.evaluate_classifier(clf, folder, extraction_type, algorithm, suffixe="_"+str(n_neighbors)+"NN_" + weights)

            except Exception as e:
                print(str(e))
                pass

            print("Ended extraction : " + extraction_type)
