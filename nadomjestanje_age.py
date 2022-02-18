"""
Prvi dio projekta: učitavaju se podaci, pravi njihov pregled i provjerava nedostaju li podaci u
pojedinim kategorijama. Nakon toga analiziraju se vrste varijabli i raspodjele podataka te se određuje što će se
napraviti s nepotpunim podacima. Kako bi se dobili što bolji rezultati, podaci će se skalirati na 2 načina,
dići u prostor viših značajki te provesti Principal Component Analysis (PCA) analizu.
"""

import pickle, shutil, os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import decomposition

project_data_win = "C:/Users/Josip/PycharmProjects/Titanic_project/project_data/"   # trebaju biti input i output folderi
project_data_linux = "//home/josip/PycharmProjects/Titanic_project/project_data/"


project_data = project_data_win

input_data, output_data = project_data+ "input/", project_data+ "output/"

if os.path.exists(output_data+"A_preprocessing") == False:        # pravi folder ako ne postoji
    os.mkdir(output_data+"A_preprocessing")
if os.path.exists(output_data+"A_preprocessing/category_diagrams") == False:        # pravi folder ako ne postoji
    os.mkdir(output_data+"A_preprocessing/category_diagrams/")


always_same_data = False  # određuje hoće li se korisitit random dijeljenje podataka za train set, kraj skripte

# Učitavanje podataka u DataFrame
train_df = pd.read_csv(input_data+'train.csv')
test_df = pd.read_csv(input_data+'test.csv')
all_df = train_df.append(test_df)

##################################        1. Statistika i opći pregled        #########################################

# Pregled varijabli i eventualni nedostatak podataka:
test_df.Fare.fillna(test_df.Fare.mean(), inplace=True)

# print(train_df.info())
# print(test_df.info())






"""
Pregledom svih varijabli došlo se do sljedećih zaključaka:
Kategoričke varijable: Survived, Sex, Embarked
Ordinalne: Pclass
Numeričke: Age, Fare
Diskretne: SibSp, Parch
Age i Cabin imaju nepotpune podatke u train i test setu: nadopunjavati ili izbaciti?
Pretpostavke:
Žene, djeca i putnici 1. klase imaju veću vjerojatnost preživljavanja
"""




###################################        2. Transformiranje podataka        #########################################














# Transformacija kontinuiranih varijabli u kategoričke
def transform_data(input_df, set_for_train):

    # Age se dijeli po osobnim, realnim kategorijama, (5 starosnih kategorija)
    input_df["Age"] = pd.cut(input_df["Age"],bins=[0,5,18,35,60,300],
                            labels=[1,2,3,4,5])


    # if set_for_train == True:
    f = input_df.Fare
    for i in f:
        # uvjet = (i < 0)     # nema
        # uvjet = (i >600)    # nema
        uvjet = (i in [0,10,30,50,80,120,300,600])
        # if uvjet == True:
        #     print(i)

    # Fare se dijeli osvisno o distribuciji na temelju procjene i razlike iz grafa (7 kategorija)
    # input_df["Fare"] = pd.cut(input_df["Fare"],bins=[0,10,30,50,80,120,300,600],
    #                          labels=[1,2,3,4,5,6,7])


    # Sex i Embarked se jednostavno preslikavaju
    input_df["Sex"] = input_df["Sex"].map({"male":0, "female":1})
    input_df["Embarked"] = input_df["Embarked"].map({"C":1, "Q":2, "S":3})

    # Izbacivanje nepotrebnih podataka
    input_df = input_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])       # izbacivanje nepotrebnih kolona





    # Dijeljenje train i test seta
    if set_for_train == True:
        input_df = input_df.dropna()                        # BEZ OVOG NE RADI
        # X_data = input_df.drop(columns=["Survived"])        # Za train set treba izbaciti "Survived" kategoriju
        X_data = input_df
        Y_data = input_df["Survived"]
        # print(X_data.info())

    elif set_for_train == False:
        X_data = input_df                   # Za test set ne postoji "Survived" kategorija
        Y_data = None

    return [X_data, Y_data]


# Transformirani podaci gdje su kontinuirane varijable prebačene u kategoriče
train_filtered_data = transform_data(train_df, set_for_train=True)
X_train = train_filtered_data[0]
Y_train = train_filtered_data[1]

train_filtered_data = transform_data(test_df, set_for_train=False)
X_test = train_filtered_data[0]

# X_test.Fare.fillna(X_test.Fare.mean(), inplace=True)





from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

y_godine = X_train["Age"]
print(y_godine)
x_ostalo = X_train.drop(columns=["Age"])

x_1, x_2, y_1, y_2 = train_test_split(x_ostalo, y_godine, random_state=True, test_size=0.1)
def kNN_f(n_neighb):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighb)
    knn_model.fit(x_1, y_1)
    acc_train = round(knn_model.score(x_1, y_1) * 100, 2)
    prediction = knn_model.predict(x_2)
    acc_valid = round(accuracy_score(prediction, y_2) * 100, 2)
    # print(n_neighb, acc_valid)
    # print(prediction)
    return prediction




# n_neigh_range = [i for i in range(1,21)]
# for i in n_neigh_range:
#     kNN_f(i)

# kNN_f(12)
# df.loc[df['First Season'] > 1990, 'First Season'] = 1


train_df["nan"] = [math.isnan(i) for i in train_df["Age"]]
nan_df = train_df[train_df["nan"]==True]
not_nan_df = train_df[train_df["nan"]==False]

nan_df = nan_df.drop(columns=["nan", "Age", "Cabin", "Ticket", "PassengerId", "Name"])
not_nan_df = not_nan_df.drop(columns=["nan", "Cabin", "Ticket", "PassengerId", "Name"])


knn_model = KNeighborsClassifier(n_neighbors=15)
knn_model.fit(x_ostalo, y_godine)
nan_df["Age"] = knn_model.predict(nan_df)

train_X_final = pd.concat([nan_df, not_nan_df], ignore_index=False).sort_index()





# print(x_ostalo)












# b = train_df[train_df["PassengerId"] > 888]

# df1 = df[df.isna(axis=1).any()]

# b = train_df['Age'].isnull().values.any()     # provjerava ima li i jedna vrijednost null

# df2 = pd.DataFrame(data=None, columns=df1.columns, index=df1.index)   kopiranje praznog dataframe-a

 # math.isnan(x):