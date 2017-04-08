from sklearn import neighbors
import SDK as sdk
import os
import personal_settings

path = personal_settings.PATH
algorithm = os.path.basename(__file__).split(".py")[0]



#execute for all datasets:
for extraction_type in os.listdir(path):
    print("Starting new classification. Extraction method : " + extraction_type)
    folder = path + extraction_type + "/"
    for n in range(10, 31):
        print("Starting new classification. Extraction method : " + extraction_type)
        print("NN = " + str(n))

        try:
            clf = neighbors.KNeighborsClassifier(n_neighbors = n, n_jobs = -1)
            sdk.evaluate_classifier(clf, folder, extraction_type, algorithm, suffixe=str(n) + "NN")

        except Exception as e:
            print(str(e))
            pass
    print("Ended extraction : " + extraction_type)


# execute for testing and evaluating the best parameters
# test_extraction_type = "MSD-JMIRMFCCS"
# algorithm = "knn_test"
# test_folder = path + test_extraction_type + "/"
#
# for n in range(12, 21):
#     print("Starting new classification. Extraction method : " + test_extraction_type)
#     print("NN = " + str(n))
#
#     clf = neighbors.KNeighborsClassifier(n_neighbors=n)
#     sdk.evaluate_classifier(clf, test_folder, test_extraction_type, algorithm, suffixe=str(n)+"NN")
#
#     print("Ended extraction : " + test_extraction_type)
