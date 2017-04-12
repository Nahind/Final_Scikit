from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import SDK as sdk
import os
import personal_settings

path = personal_settings.PATH
algorithm = os.path.basename(__file__).split(".py")[0]
datasets = personal_settings.BEST_DATASETS


#execute for all datasets:
def neural_on_all_datasets():
    scaler = StandardScaler()
    # for extraction_type in os.listdir(path):
    for extraction_type in datasets:
        print("Starting new classification. Extraction method : " + extraction_type)
        folder = path + extraction_type + "/"

        try:
            training, validation = sdk.load_dataset_from_folder(folder, extraction_type)
            # Normalize training & validation sets
            print("Start normalizing data")
            scaler.fit(training.data)
            training.data = scaler.transform(training.data)
            print("Training data has been normalized")
            validation.data = scaler.transform(validation.data)
            print("Validation data has been normalized")
            # Normalize data
            print("Start balancing training data")
            xs, ys = sdk.balanced_subsample(training.data, training.target)
            # Build MLP model
            print("Start training MLP Classifier")
            hidden_layer_size = 200
            clf = MLPClassifier(verbose=True, hidden_layer_sizes=hidden_layer_size)
            model = clf.fit(xs, ys)
            print("Training has ended")
            # Save MLP model
            sdk.save_model(model, extraction_type, algorithm)
            # Evaluation model
            print("Start predicting validation set")
            y_pred = model.predict(validation.data)
            # Save Evaluation report
            sdk.save_classification_report(validation, extraction_type, y_pred, algorithm, suffixe="__real_no_early_stop_balanced_data_hidden_"+str(hidden_layer_size)+"_layers")

        except Exception as e:
            print(str(e))
            pass
        print("Ended extraction : " + extraction_type)


# apply neural network algorithm on all datasets
neural_on_all_datasets()
