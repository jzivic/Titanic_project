import pickle, shutil, os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import decomposition


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



project_data = "C:/Users/Josip/PycharmProjects/Titanic_project/project_data/"   # trebaju biti input i output folderi
input_data, output_data = project_data+ "input/", project_data+ "output/"

if os.path.exists(output_data+"A_preprocessing") == False:        # pravi folder ako ne postoji
    os.mkdir(output_data+"A_preprocessing")
if os.path.exists(output_data+"A_preprocessing/category_diagrams") == False:        # pravi folder ako ne postoji
    os.mkdir(output_data+"A_preprocessing/category_diagrams/")


always_same_data = True  # određuje hoće li se korisitit random dijeljenje podataka za train set, kraj skripte

# Učitavanje podataka u DataFrame
train_df = pd.read_csv(input_data+'train.csv')
test_df = pd.read_csv(input_data+'test.csv')
all_df = train_df.append(test_df)

test_df.Fare.fillna(test_df.Fare.mean(), inplace=True)  # nadomjestanje jedinog podatka koji nedostaje







# Transformacija kontinuiranih varijabli u kategoričke
def transform_data(input_df, set_for_train):

    # Sex i Embarked se jednostavno preslikavaju
    input_df["Sex"] = input_df["Sex"].map({"male":0, "female":1})
    input_df["Embarked"] = input_df["Embarked"].map({"C":1, "Q":2, "S":3})

    input_df["Age"] = pd.cut(input_df["Age"],bins=[0,5,18,35,60,300],
                            labels=[1,2,3,4,5])

    # Izbacivanje nepotrebnih podataka
    input_df = input_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])       # izbacivanje nepotrebnih kolona

    # filtriranje nan podataka za Age kategoriju
    input_df["nan_Age"] = [math.isnan(i) for i in input_df["Age"]]
    nan_df = input_df[input_df["nan_Age"]==True]
    not_nan_df = input_df[input_df["nan_Age"]==False]

    nan_df = nan_df.drop(columns=["nan_Age"])
    not_nan_df = not_nan_df.drop(columns=["nan_Age"])


    # # Dijeljenje train i test seta
    if set_for_train == True:
        X_data = input_df.drop(columns=["Survived"])        # Za train set treba izbaciti "Survived" kategoriju
        Y_data = input_df["Survived"]
        not_nan_df = not_nan_df.drop(columns=["Survived"])
        not_nan_df.Embarked.fillna(not_nan_df.Embarked.mean(), inplace=True)


    elif set_for_train == False:
        X_data = input_df                   # Za test set ne postoji "Survived" kategorija
        Y_data = None


    not_nan_df = not_nan_df.dropna()

    # print(not_nan_df.info())


    X_data_dict = {"All":X_data,"nan_df":nan_df,"not_nan_df":not_nan_df, }
    # return [X_data, Y_data, nan_df, not_nan_df]
    return [X_data_dict, Y_data]


train_filtered_data = transform_data(test_df, set_for_train=False)
# X_test = train_filtered_data[0]
X_test = train_filtered_data[0]["All"]



train_filtered_data = transform_data(train_df, set_for_train=True)
# X_train = train_filtered_data[0]
# Y_train = train_filtered_data[1]
X_train = train_filtered_data[0]["All"]
nan_df = train_filtered_data[0]["nan_df"]
not_nan_df = train_filtered_data[0]["not_nan_df"]

Y_train = train_filtered_data[1]



# print(not_nan_df.info())


# dio za učenje algoritma za predviđanje godina
# y_godine = X_train["Age"]
# x_ostalo = X_train.drop(columns=["Age"])
# x_1, x_2, y_1, y_2 = train_test_split(x_ostalo, y_godine, random_state=True, test_size=0.1)

x_ostalo = not_nan_df.drop(columns=["Age"])
y_godine = not_nan_df["Age"]


for i in x_ostalo:
    print(x_ostalo[i])


knn_model = KNeighborsClassifier(n_neighbors=15)
knn_model.fit(x_ostalo, y_godine)
nan_df["Age"] = knn_model.predict(nan_df)

# print(y_godine)








