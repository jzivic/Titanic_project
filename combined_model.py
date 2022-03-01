"""
Zadnji dio projekta: Cilj je naći konačno predviđanje preživljavanja putnika.
Prvo će se usporediti nekoliko najboljih modela te naći prosjek njhovih previđanja.
Rješenje će se zapisati u csv file.
"""

from Preprocessing import divided_train_data, all_X_test_data
from Preprocessing import input_data, output_data

import pickle, shutil, os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


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


# Funkcije za svaki algoritam, odabire se algoritam, korišteni podaci i hiperparametri
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

# pred_SVC = chosen_model_SVC(data_set="poly4_X")



def chosen_model_kNN(data_set):
    knn_model = KNeighborsClassifier(n_neighbors=5)
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
pred_decision_tree = chosen_model_decision_tree(data_set="X")


def chosen_model_random_forest(data_set):
    random_forest_model = RandomForestClassifier(n_estimators=100)
    random_forest_model.fit(divided_train_data["X_all_data"][data_set], divided_train_data["Y_all_data"][data_set])
    prediction = random_forest_model.predict(all_X_test_data[data_set])
    # print(prediction)
    return prediction
# pred_random_forest = chosen_model_random_forest(data_set="poly4_X")



def chosen_model_gaussian_NB(data_set):
    gaussian_model = GaussianNB()
    gaussian_model.fit(divided_train_data["X_all_data"][data_set], divided_train_data["Y_all_data"][data_set])
    prediction = gaussian_model.predict(all_X_test_data[data_set])
    return prediction

pred_gaussian_NB = chosen_model_gaussian_NB(data_set="X")




# Lista rješenja za sve korištene algoritme
# all_predictions = [pred_LogReg, pred_SVC, pred_kNN, pred_decision_tree, pred_random_forest]
# all_predictions = [pred_LogReg, pred_kNN, pred_decision_tree, pred_random_forest]
all_predictions = [pred_LogReg, pred_kNN, pred_decision_tree]

all_predictions = [pred_gaussian_NB]


combined_predictions = list(zip(*all_predictions))       # grupiranje predviđanja po putniku za svaki primjer
prediction = [round(sum(i)/len(i)) for i in combined_predictions]   # prosjek predviđanja
passangerId = list(range(892,1310))

survived_df = pd.DataFrame({"Survived":prediction}, index=passangerId)  # DF
survived_df.to_csv(final_prediction + "final_prediction.csv")    # spremanje podataka u csv


# Dodano zbog jednostavnosti da se promijeni prva linija koja se zbog df ne može točno zapisati kako se traži
with open(final_prediction + "final_prediction.csv", "r") as f:
    lines = f.readlines()
lines[0] = "PassengerId,Survived\n"

with open(final_prediction + "final_prediction.csv", "w") as f:
    lines = f.writelines(lines)




"""
Zaključak:
- Na temelju uspoređivanja nekoliko predikcija, dolazi se do zaključka da je model tuniran do svoje granice: 
    To potvrđuje činjenica da više modela s tuniraim parametrima ima sličnu točnost koja bi objektivno 
    mogla biti bolja na temelju rezultata drugih ljudi. 
- Bolji rezultati bi se mogli dobiti detaljnijim interveniranjem na samim podacima: dodavanjem nekih značajki 
    ili  njihovom daljnom kombinacijom i analizom. 
"""