from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import SDK as sdk
import personal_settings

###
# Variable definition that we will use
###

# path where all data files are saved
path = personal_settings.PATH

# define classifiers with best parameters
clf_adaboost = AdaBoostClassifier(n_estimators=500)
clf_naive = GaussianNB()
clf_gradient = GradientBoostingClassifier(verbose=20, n_estimators=300)
clf_mlp = MLPClassifier(verbose=True, hidden_layer_sizes=200)
clf_svm = SVC()
clf_knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1, weights='distance')
clf_random_f = RandomForestClassifier(n_jobs=-1, verbose=20, n_estimators=100)
clf_one_vs_rest = OneVsRestClassifier(clf_random_f, n_jobs=-1)
clf_voting = VotingClassifier(estimators=[('mlp', clf_mlp), ('nb', clf_naive), ('knn', clf_knn)], voting='soft',
                                weights=[2, 1, 2], n_jobs=-1)
clf_d_tree = DecisionTreeClassifier()


###
# Script that we are testing
###
