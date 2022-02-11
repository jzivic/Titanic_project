
"""
Treba otkriti koji podaci najbolje odgovaraju klasifikfatoru
"""

from Preprocessing import input_data, output_data
# from Preprocessing import Y_train
# from Preprocessing import divided_train_data, all_X_test_data
from sklearn.metrics import accuracy_score

import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


with open(input_data+'/divided_train_data.pickle', 'rb') as f_X_train:
    divided_train_data = pickle.load(f_X_train)
with open(input_data+'/all_X_test_data.pickle', 'rb') as f_test:
    all_X_test_data = pickle.load(f_test)


def gaussian_NB_f(X_train, Y_train, X_valid, Y_valid):
    gaussian_model = GaussianNB()
    gaussian_model.fit(X_train, Y_train)
    acc_train = round(gaussian_model.score(X_train, Y_train) * 100, 2)
    prediction = gaussian_model.predict(X_valid)
    acc_valid = round(accuracy_score(prediction, Y_valid) * 100, 2)

    return [acc_train, acc_valid]




def random_forest_f(X_train, Y_train, X_valid, Y_valid):
    random_forest_model = RandomForestClassifier(n_estimators=100)
    random_forest_model.fit(X_train, Y_train)
    acc_train = round(random_forest_model.score(X_train, Y_train) * 100, 2)
    prediction = random_forest_model.predict(X_valid)
    acc_valid = round(accuracy_score(prediction, Y_valid) * 100, 2)
    return [acc_train, acc_valid]


def decision_tree_f(X_train, Y_train, X_valid, Y_valid):
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(X_train, Y_train)

    acc_train = round(decision_tree_model.score(X_train, Y_train) * 100, 2)
    prediction = decision_tree_model.predict(X_valid)
    acc_valid = round(accuracy_score(prediction, Y_valid) * 100, 2)

    # print(acc_train, acc_valid)
    return [acc_train, acc_valid]




for data_name in divided_train_data["X_train_data"]:
    X_train = divided_train_data["X_train_data"][data_name]
    Y_train = divided_train_data["Y_train_data"][data_name]
    X_valid = divided_train_data["X_valid_data"][data_name]
    Y_valid = divided_train_data["Y_valid_data"][data_name]

    print(data_name,
          decision_tree_f(X_train, Y_train, X_valid, Y_valid),
          random_forest_f(X_train, Y_train, X_valid, Y_valid),
          # gaussian_NB_f(X_train, Y_train, X_valid, Y_valid)
          )



"""
Najbolje je uzeti obične podatke i koristiti decision tree algoritam
"""