from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import SDK as sdk
import personal_settings

path = personal_settings.PATH
algorithm = "adaboost_nn"
suffixe = "_neural_net"


datasets = personal_settings.SMALL_DATASETS

for extraction_type in datasets:
    print("Starting new classification. Extraction method : " + extraction_type)
    folder = path + extraction_type + "/"

    try :
        scaler = StandardScaler()
        training, validation = sdk.load_dataset_from_folder(folder, extraction_type)
        # Normalize training & validation sets
        print("Start normalizing data")
        scaler.fit(training.data)
        training.data = scaler.transform(training.data)
        print("Training data has been normalized")
        validation.data = scaler.transform(validation.data)
        print("Validation data has been normalized")
        # Build MLP model
        print("Start training MLP Classifier")
        clf = MLPClassifier(verbose=True, hidden_layer_sizes=300, early_stopping=True, validation_fraction=0.2)
        # Create and fit an AdaBoosted MLP
        bdt = AdaBoostClassifier(clf, algorithm="SAMME", n_estimators=500)
        model = clf.fit(training.data, training.target)
        print("Training has ended")
        # Save MLP model
        sdk.save_model(model, extraction_type, algorithm)
        # Evaluation model
        print("Start predicting validation set")
        y_pred = model.predict(validation.data)
        # Save Evaluation report
        sdk.save_classification_report(validation, extraction_type, y_pred, algorithm, suffixe=suffixe)
    
    except Exception as e:
        print(str(e))
        pass

    print("Ended extraction : " + extraction_type)
