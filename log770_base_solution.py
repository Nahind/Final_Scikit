###
# This is the implementation of the base solution classifier
# To get that solution working you have to change the attribute folder_of_datasets
# it should point to the folder containing the train & the validation files
###

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import SDK as sdk


folder_of_datasets = "/home/nandane/Documents/Cours_ETS_MTL/LOG770_Intelligence_machine/LAB4/DEV_PREPARED/MSD-JMIRMFCCS"
algorithm = "mlp"
extraction_type = "MSD-MARSYAS"

# To Normalize data
scaler = StandardScaler()

print("Starting new classification. Extraction method : " + extraction_type)
folder_train_valid = folder_of_datasets + "/"

try:
    training, validation = sdk.load_dataset_from_folder(folder_train_valid, extraction_type)
    # Normalize training & validation sets
    print("Start normalizing data")
    scaler.fit(training.data)
    training.data = scaler.transform(training.data)
    print("Training data has been normalized")
    validation.data = scaler.transform(validation.data)
    print("Validation data has been normalized")
    # Build MLP model
    print("Start training MLP Classifier")
    hidden_layer_size = 300
    clf = MLPClassifier(verbose=True, hidden_layer_sizes=hidden_layer_size, early_stopping=True)
    model = clf.fit(training.data, training.target)
    print("Training has ended")
    # Save MLP model
    sdk.save_model(model, extraction_type, algorithm)
    # Evaluation model
    print("Start predicting validation set")
    y_pred = model.predict(validation.data)
    # Save Evaluation report
    sdk.save_classification_report(validation, extraction_type, y_pred, algorithm,
                                   suffixe="early_stop_"+str(hidden_layer_size)+"_layers")

except Exception as e:
    print(str(e))
    pass

print("Ended extraction : " + extraction_type)

