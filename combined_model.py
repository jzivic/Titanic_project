
from Preprocessing import Y_train
from Preprocessing import divided_train_data, all_X_test_data

from Preprocessing import input_data, output_data

import pickle, shutil, os, openpyxl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


final_prediction = output_data+"final_prediction/"

with open(input_data+'/divided_train_data.pickle', 'rb') as f_X_train:
    divided_train_data = pickle.load(f_X_train)
with open(input_data+'/all_X_test_data.pickle', 'rb') as f_test:
    all_X_test_data = pickle.load(f_test)

try:
    shutil.rmtree(final_prediction)
except:
    FileNotFoundError
os.mkdir(final_prediction)



def chosen_model_LogReg(data_set):
    # logreg_model = LogisticRegression(C=10e-5, max_iter=1e7)        # Broj iteracija povećan zbog poly podataka
    logreg_model = LogisticRegression(C=10**2)        # Broj iteracija povećan zbog poly podataka
    # logreg_model.fit(divided_train_data["X_all_data"][data_set], divided_train_data["Y_all_data"][data_set])
    logreg_model.fit(divided_train_data["X_train_data"][data_set], divided_train_data["Y_train_data"][data_set])
    prediction = logreg_model.predict(all_X_test_data[data_set])
    # print(prediction)
    return prediction
pred_LogReg = chosen_model_LogReg(data_set="X")



def chosen_model_SVC(data_set):
    svc_model = SVC(C=0.25, kernel="rbf", gamma=3e-6)
    svc_model.fit(divided_train_data["X_all_data"][data_set], divided_train_data["Y_all_data"][data_set])
    prediction = svc_model.predict(all_X_test_data[data_set])
    # print(prediction)
    return prediction

pred_SVC = chosen_model_SVC(data_set="poly4_X")



def chosen_model_kNN(data_set):
    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(divided_train_data["X_all_data"][data_set], divided_train_data["Y_all_data"][data_set])
    prediction = knn_model.predict(all_X_test_data[data_set])
    # print(prediction)
    return prediction
pred_kNN = chosen_model_kNN(data_set="scal_MM_X")



def chosen_model_decision_tree(data_set):
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(divided_train_data["X_all_data"][data_set], divided_train_data["Y_all_data"][data_set])
    prediction = decision_tree_model.predict(all_X_test_data[data_set])
    # print(prediction)
    return prediction
pred_decision_tree = chosen_model_decision_tree(data_set="poly4_X")


def chosen_model_random_forest(data_set):
    random_forest_model = RandomForestClassifier(n_estimators=100)
    random_forest_model.fit(divided_train_data["X_all_data"][data_set], divided_train_data["Y_all_data"][data_set])
    prediction = random_forest_model.predict(all_X_test_data[data_set])
    # print(prediction)
    return prediction
pred_random_forest = chosen_model_random_forest(data_set="poly4_X")


# all_predictions = [pred_LogReg, pred_SVC, pred_kNN, pred_decision_tree, pred_random_forest]
all_predictions = [pred_LogReg, pred_kNN, pred_decision_tree, pred_random_forest]
# all_predictions = [pred_LogReg, pred_SVC, pred_kNN]

all_predictions = [pred_LogReg]

combined_predictions = list(zip(*all_predictions))
prediction = [round(sum(i)/len(i)) for i in combined_predictions]
passangerId = list(range(892,1310))



survived_df = pd.DataFrame({"Survived":prediction}, index=passangerId)
survived_df.to_csv(final_prediction + "rjesenje_final2.csv")
survived_df.to_excel(final_prediction + "rjesenje_final.xlsx")



#            PassengerId,Survived


