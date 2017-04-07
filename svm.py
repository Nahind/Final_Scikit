import personal_settings
import os
import SDK as sdk

from sklearn.svm import SVC
from sklearn import metrics

path = personal_settings.PATH
algorithm = os.path.basename(__file__).split(".py")[0]

training, validation, extraction_type = sdk.get_test_dataset(path)
clf = SVC()
model = clf.fit(training.data, training.target)
print("Training has ended")
# Save MLP model
sdk.save_model(model, extraction_type, algorithm)
# Evaluation model
print("Start predicting validation set")
y_pred = model.predict(validation.data)
# Save Evaluation report
sdk.save_classification_report(validation, extraction_type, y_pred, algorithm)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):

    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    print(metrics.classification_report(validation.target, y_pred))
    print()