"""
Prvi dio projekta: učitavaju se podaci, pravi njihov pregled i provjerava nedostaju li podaci u
pojedinim kategorijama. Nakon toga analiziraju se vrste varijabli i raspodjele podataka te se određuje što će se
napraviti s nepotpunim podacima. Kako bi se dobili što bolji rezultati, podaci će se skalirati, dići u prostor
 viših značajki te provesti Principal Component Analysis (PCA) analizu.
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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ovako trebaju biti postavljeni folderi za automatsko spremanje input i output podataka
project_data = "C:/Users/Josip/PycharmProjects/Titanic_project/project_data/"

input_data, output_data = project_data+ "input/", project_data+ "output/"

if os.path.exists(output_data+"A_preprocessing") == False:        # kreira folder ako ne postoji
    os.mkdir(output_data+"A_preprocessing")
if os.path.exists(output_data+"A_preprocessing/category_diagrams") == False:        # kreira folder ako ne postoji
    os.mkdir(output_data+"A_preprocessing/category_diagrams/")


always_same_data = False  # određuje hoće li se korisitit random opcija za dijeljenje podataka train seta
# ili će biti uvijek isti podaci. Dijeljenje seta se nalazi na dnu skripte

# Učitavanje podataka u DataFrame
train_df = pd.read_csv(input_data+'train.csv')
test_df = pd.read_csv(input_data+'test.csv')

##################################        1. Statistika i opći pregled        #########################################

# Pregled varijabli i eventualni nedostatak podataka:
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



# Opisna funkcija koja daje opći pregled i grafove preživljavanja po pojedinoj kateogriji
def survived_in_category(category):
    wanted_data = train_df[[category, 'Survived']]       # podaci željene kategorije i preživljavanja
    number_in_category = wanted_data.groupby(category).size()     # broj ljudi podijeljen po kategoriji

    survived_people = wanted_data[wanted_data["Survived"] == 1]       # preživjeli iz ukupnog popisa
    died_people = wanted_data[wanted_data["Survived"] == 0]           # umrli

    # sortiran udio preživjelih u svakoj kateogoriji
    survived_rate = wanted_data.\
        groupby([category], as_index=False).mean()\
        .sort_values(by='Survived', ascending=False)

    # brisanje foldera ako postoji i stvaranje praznog
    try:
        shutil.rmtree(output_data+"A_preprocessing/category_diagrams/"+category)
    except:
        FileNotFoundError
    os.mkdir(output_data+"A_preprocessing/category_diagrams/"+category)

    # Ukupan broj ljudi po kategoriji
    def overall_number():
        print("Ukupan broj ljudi u kategoriji: " + category)
        print(number_in_category)
        fig = plt.gcf()
        plt.title("Overall people in category:  " + category)
        plt.ylabel("Number of people")
        plt.xlabel("Category")
        plt.bar(survived_rate[category], number_in_category)
        plt.draw()
        plt.savefig(output_data+"A_preprocessing/category_diagrams/"+category+
                    "/Overall people in "+ category+".png", dpi=300)
        plt.clf()

    # Postotak preživjelih ljudi po kategoriji
    def survival_percentage():
        print("\nSurvival percentage in category: " + category)
        print(survived_rate)
        fig = plt.gcf()
        plt.title("Survival percentage in category : " + category)
        plt.ylabel("Survival percentage")
        plt.xlabel("Category")
        plt.bar(survived_rate[category], survived_rate["Survived"]*100)
        plt.draw()
        plt.savefig(output_data+"A_preprocessing/category_diagrams/"+category+
                    "/Survival percentage in "+ category+".png", dpi=300)
        plt.clf()

    # Odnos brojeva preživjelih i umrlih ljudi po kategoriji (isti grafovi, drugi način)
    def survived_vs_died():
        fig = plt.gcf()
        plt.title("Survived and died ratio:  " + category)
        plt.ylabel("Number of people")
        plt.xlabel(category)
        plt.hist(died_people[category], bins=10, color="black", edgecolor="red",  alpha=0.5,  label="died")
        plt.hist(survived_people[category], bins=10, color="blue", edgecolor="red", alpha=0.5, label="survived")
        plt.legend()
        plt.draw()
        plt.savefig(output_data+"A_preprocessing/category_diagrams/"+category+
                    "/Survived and died ratio in "+ category+".png", dpi=300)
        plt.clf()

    try:        # koristi se jer se ordinalne varijable ne mogu plotati
        overall_number()
        survival_percentage()
        survived_vs_died()
    except:
        pass


# survived_in_category("Pclass")
# survived_in_category("Sex")
# survived_in_category("Age")
# survived_in_category("SibSp")
# survived_in_category("Parch")
# survived_in_category("Fare")
# survived_in_category("Embarked")



# Funkcija za plotanje 2D grafa: 2 kategorije i preživaljanje
def dots_2D(category_1, category_2):

    try:
        shutil.rmtree(output_data+"A_preprocessing/2D_diagrams/")
    except:
        FileNotFoundError
    os.mkdir(output_data+"A_preprocessing/2D_diagrams/")

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.grid(True)
    plt.xticks(list(range(0, 100, 2)))
    sns.swarmplot(y=category_1, x=category_2, hue="Survived", data=train_df)
    plt.legend()
    plt.draw()
    plt.savefig(output_data+"A_preprocessing/2D_diagrams/"+category_1+"_"+category_2+".png", dpi=300)
    plt.clf()

# dots_2D(category_1="Sex", category_2="Age")



"""
Zaključci:
Preživljavanje je najviše vezano uz: Sex, Age, Pclass.
Također je manje vezano uz Embarked, Fare, Parch, SibSp.
Age i Fare treba transformirati u kategoričku varijablu  i odrediti raspone.

PassangerId, Name i Ticket se izbacuju jer su slučajne varijabla i ne koreliraju s preživljavanjem.
Cabin se izacuje jer za većinu putnika ne postoji podatak.
Age je bitan prediktor te se nadopunjuje za train i test set kNN algoritmom. U Fare kategoriji nedostaju samo
    2 podatka te će se nadopuniti srednjom vrijednosti kategorije.
"""

###################################        2. Transformiranje podataka        #########################################





# Funkcija koja nadopunjuje Age podatke koji nedostaju
def transform_data(input_df, set_for_train):
    # Izbacivanje nepotrebnih podataka
    input_df = input_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])       # izbacivanje nepotrebnih kolona

    # Sex i Embarked se jednostavno preslikavaju
    input_df["Sex"] = input_df["Sex"].map({"male":0, "female":1})        # mapiranje zbog nemogućnosti rada sa stringom
    input_df["Embarked"] = input_df["Embarked"].map({"C":1, "Q":2, "S":3})

    # nadopunjavanje podataka koji fale srednjom vrijenosti kategorije
    input_df["Embarked"].fillna(input_df["Embarked"].mean(), inplace=True)
    input_df["Fare"].fillna(input_df["Fare"].mean(), inplace=True)

    input_df["Age"] = pd.cut(input_df["Age"],bins=[0,5,18,35,60,300],    # dijeljenje Aege kategorije u realne okvire
                            labels=[1,2,3,4,5])

    # filtriranje nan podataka u Age kategoriji
    input_df["nan_Age"] = [math.isnan(i) for i in input_df["Age"]]      # nova kolona u input_df, bool postoji li Age
    nan_df = input_df[input_df["nan_Age"]==True]                        # novi DF za podatke gdje nema podataka za Age
    not_nan_df = input_df[input_df["nan_Age"]==False]                   # gdje postoji Age

    nan_df = nan_df.drop(columns=["nan_Age", "Age"])                    # postaje nepotrebno i briše se
    not_nan_df = not_nan_df.drop(columns=["nan_Age"])   

    y_Age = not_nan_df["Age"]
    x_without_Age = not_nan_df.drop(columns=["Age"])
    x_1, x_2, y_1, y_2 = train_test_split(x_without_Age, y_Age, random_state=True, test_size=0.1)


    def kNN_f(n_neigh):
        knn_model = KNeighborsClassifier(n_neighbors=n_neigh)
        knn_model.fit(x_1, y_1)
        acc_train = round(knn_model.score(x_1, y_1) * 100, 2)
        prediction = knn_model.predict(x_2)
        acc_valid = round(accuracy_score(prediction, y_2) *100,2)
        print(acc_train, acc_valid)
        return prediction

    # ispisivanje točnosti train i validation seta za određen broj susjeda
    # for n_n in range(1,36):
    #     acc = kNN_f(n_n)
    # uzima se 18 ili 30 susjeda, 18 bolje odgovara train a 30 test setu

    # konačno nadopunjavanje godina
    knn_model = KNeighborsClassifier(n_neighbors=18)
    knn_model.fit(x_without_Age, y_Age)
    prediction_age = knn_model.predict(nan_df)
    nan_df["Age"] = prediction_age                  # nadopunjavanje godina

    filled_data = pd.concat([nan_df, not_nan_df], ignore_index=False).sort_index()  # spajanje svih vrijednosti

    if set_for_train == True:
        X_data = filled_data.drop(columns=["Survived"])        # Za train set treba izbaciti "Survived" kategoriju
        Y_data = filled_data["Survived"]

    elif set_for_train == False:
        X_data = filled_data                   # Za test set ne postoji "Survived" kategorija
        Y_data = None

    return [X_data, Y_data]




# Transformirani podaci gdje su kontinuirane varijable prebačene u kategoričke
train_filtered_data = transform_data(train_df, set_for_train=True)
X_train = train_filtered_data[0]
Y_train = train_filtered_data[1]

train_filtered_data = transform_data(test_df, set_for_train=False)
X_test = train_filtered_data[0]


###################################        3. Manipulacije podacima        #########################################

"""
Kako bi se dobila što veća točnost modela, podaci će se transformirati u nekoliko kategorija te će se vidjeti
koji imaju najbolju točnost. Transformacije su Standardno i MinMax skaliranje, dizanje u viši prostor značajki za 
različit broj razina te PCA analiza odnosno redukcija značajki. 
"""

# A: skaliranje podataka: Standardno i MinMax
def scalling_data(type_of_scaling):
    # U slučaju pogrešne ključne riječi skaliranja dobiva se error:
    assert type_of_scaling == StandardScaler or type_of_scaling == MinMaxScaler, "Skaliranje mora biti Std ili MinMax"

    scal_data_test = pd.DataFrame()
    categories = [categ for categ in X_train]
    scal_data_train = pd.DataFrame()
    if type_of_scaling == StandardScaler:
        scaler_model = StandardScaler()
        scal_data_train[categories] = scaler_model.fit_transform(X_train[categories])   # skaliranje svake kategorije
        scal_data_test[categories] = scaler_model.transform(X_test[categories])

    elif type_of_scaling == MinMaxScaler:
        scaler_model = MinMaxScaler()
        scal_data_train[categories] = scaler_model.fit_transform(X_train[categories])
        scal_data_test[categories] = scaler_model.transform(X_test[categories])

    return [scal_data_train, scal_data_test]


scal_data = scalling_data(StandardScaler)   # Podaci s transformacijom standardiziranog skaliranja
scal_std_X_train = scal_data[0]
scal_std_X_test = scal_data[1]

scal_data = scalling_data(MinMaxScaler)     # Podaci s MinMax skaliranjem
scal_MM_X_train = scal_data[0]
scal_MM_X_test = scal_data[1]


# B: dizanje matrice dizajna u prostor više dimenzije (kombinacija postojećih kategorija)
def polynomial_features_scaling(deg):
    p_X_train = sklearn.preprocessing.PolynomialFeatures(degree=deg).fit_transform(X_train)
    p_X_test = sklearn.preprocessing.PolynomialFeatures(degree=deg).fit_transform(X_test)
    poly_X_train = pd.DataFrame(p_X_train)
    poly_X_test = pd.DataFrame(p_X_test)

    return [poly_X_train, poly_X_test]


poly3_data = polynomial_features_scaling(deg=3)         # Podaci dignuti za 3 stupnja
poly3_X_train = poly3_data[0]
poly3_X_test = poly3_data[1]

poly4_data = polynomial_features_scaling(deg=4)         # Podaci dignuti za 4 stupnja
poly4_X_train = poly4_data[0]
poly4_X_test = poly4_data[1]



# C: Principal component analysis: redukcije značajki i utjecaj na vrijeme/točnost modela
# Provjera točnosti podataka nakon redukcije: bitno je koristiti već skalirane podatke
def PCA_f(n_components):
    assert n_components <= len(X_train.columns), "Broj komponenata je prevelik!"  # max komponenata koje mogu ostati
    pca = decomposition.PCA(n_components=n_components)
    pca_X_train = pca.fit_transform(scal_std_X_train)
    pca_X_test = pca.fit_transform(scal_std_X_test)

    # Singular Value Decomposition
    U, sigma, V = np.linalg.svd(pca_X_train)  # lijeva ort. matrica , matrica vlastitih vrijednosti, desna ort. matrica
    sum_current_sigma = sum([sigma[i] for i in range(n_components)])
    sum_whole_sigma = sum(np.linalg.svd(decomposition.PCA(n_components=7).fit_transform(scal_std_X_train))[1])
    data_preservation = (sum_current_sigma/sum_whole_sigma)         # očuvanost podataka nakon transformacije
    # print("Očuvanost podataka: ", round(data_preservation,2))

    return [pca_X_train, pca_X_test]


pca_6_X = PCA_f(6)
pca_6_X_train = pca_6_X[0] # Podaci reducirani za jednu značajku
pca_6_X_test = pca_6_X[1]
pca_5_X = PCA_f(5)
pca_5_X_train = pca_5_X[0] # Podaci reducirani za dvije značajku
pca_5_X_test = pca_5_X[1]
pca_4_X = PCA_f(4)
pca_4_X_train = pca_4_X[0] # Podaci reducirani za jednu značajku
pca_4_X_test = pca_4_X[1]


# Rječnik sa svim test podacima za predviđanje
all_X_test_data = { "X": X_test,
                    "scal_std_X": scal_std_X_test,
                    "scal_MM_X": scal_MM_X_test,
                    "poly3_X": poly3_X_test,
                    "poly4_X": poly4_X_test,
                    "pca_6_X": pca_6_X_test,
                    "pca_5_X": pca_5_X_test,
                    "pca_4_X": pca_4_X_test,
                   }


with open(input_data+'all_X_test_data.pickle', 'wb') as f_test:    # spremanje riječnika u dictdf
    pickle.dump(all_X_test_data, f_test)



# Rječnik sa svim ukupnim train podacima
all_X_train_data = {
                    "X": X_train,
                    "scal_std_X": scal_std_X_train,
                    "scal_MM_X": scal_MM_X_train,
                    # "poly3_X": poly3_X_train,
                    # "poly4_X": poly4_X_train,
                    "pca_6_X": pca_6_X_train,
                    "pca_5_X": pca_5_X_train,
                    "pca_4_X": pca_4_X_train,
                    }


# Skup za treniranje se dijeli na skup za treniranje i skup za validaciju modela
X_train_data, X_valid_data, Y_train_data, Y_valid_data, X_all_data, Y_all_data = {}, {}, {}, {}, {}, {}
for data in all_X_train_data:

    if always_same_data == True:
        X_train, X_valid, y_train, y_valid = train_test_split(all_X_train_data[data], Y_train,
                                                              random_state=True, test_size=0.1)
    elif always_same_data == False:
        X_train, X_valid, y_train, y_valid = train_test_split(all_X_train_data[data], Y_train, test_size=0.1)

    X_train_data[data] = X_train
    X_valid_data[data] = X_valid
    X_all_data[data] = all_X_train_data[data]

    Y_train_data[data] = y_train
    Y_valid_data[data] = y_valid
    Y_all_data[data] = Y_train


# Ukupni podijeljenji podaci na train i validation set
divided_train_data = {
                    "X_train_data": X_train_data,
                    "X_valid_data": X_valid_data,
                    "Y_train_data": Y_train_data,
                    "Y_valid_data": Y_valid_data,

                    "X_all_data": X_all_data,
                    "Y_all_data": Y_all_data,
                    }



# Spremanje podataka u rječnik
with open(input_data+'divided_train_data.pickle', 'wb') as f_test:    # spremanje riječnika u dictdf
    pickle.dump(divided_train_data, f_test)








print()
# DOBRE FORE

# train_df['Pclass'].value_counts(sort=False).plot(kind='bar')
# plt.show()

# t_f_null = train_df["Age"].isnull()#.values.any()
# index_null = [i for i in range(len(t_f_null)) if t_f_null[i] == True]

# stariji_50 = train_df[train_df["Age"] > 50]
# stariji_50 = train_df[train_df.Age > 50]      # Isti način zapisa


# mapiranje / preslikavanje
# maping = {"male":1, "female":2}
# project_data = train_df["Sex"].map(maping)




# tocnost = sklearn.metrics.accuracy_score(y_test, h)

# varijablino ime se ispisuje
# data_name = f'{variable=}'.split('=')[0]




# df.loc[df['First Season'] > 1990, 'First Season'] = 1










