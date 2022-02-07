"""
Prvi dio projekta: učitavaju se podaci, pravi njihov pregled i provjerava nedostaju li podaci u
pojedinim kategorijama. Nakon toga analiziraju se vrste varijabli i raspodjele podataka te se određuje što će se
napraviti s nepotpunim podacima. Kako bi se dobili što bolji rezultati, podaci će se skalirati na 2 načina,
dići u prostor viših značajki te provesti Principal Component Analysis (PCA) analizu.
"""

import pickle, shutil, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import decomposition

global project_path
# project_path = "C:/Users/Josip/PycharmProjects/Titanic_project/"
# output_folder = "C:/Users/Josip/PycharmProjects/Titanic_project/output_data"

project_data = "C:/Users/Josip/PycharmProjects/Titanic_project/project_data/"   # trebaju biti input i output folderi


input_data = project_data+ "input/"
output_data = project_data+ "output/"


if os.path.exists(output_data+"A_preprocessing") == False:        # pravi folder ako ne postoji
    os.mkdir(output_data+"A_preprocessing")
if os.path.exists(output_data+"A_preprocessing/category_diagrams") == False:        # pravi folder ako ne postoji
    os.mkdir(output_data+"A_preprocessing/category_diagrams/")


# Učitavanje podataka u DataFrame
train_df = pd.read_csv(input_data+'train.csv')
test_df = pd.read_csv(input_data+'test.csv')
all_df = train_df.append(test_df)

##################################        1. Statistika i opći pregled        #########################################



def dots_2D(dataset, category_1,category_2):
    fig,ax = plt.subplots(figsize=(18,5))
    ax.grid(True)
    plt.xticks(list(range(0,100,2)))
    sns.swarmplot(y=category_1, x=category_2, hue="Survived", data=train_df)
    plt.show()
# dots_2D(train_df, "Sex", "Age")



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

    try:
        overall_number()
        survival_percentage()
        survived_vs_died()
    except:
        pass


survived_in_category("Pclass")
survived_in_category("Sex")
survived_in_category("Age")
survived_in_category("SibSp")
survived_in_category("Parch")
survived_in_category("Fare")
survived_in_category("Embarked")



# Funkcija za plotanje 2D grafa: 2 kategorije i preživaljanje
def dots_2D(category_1, category_2):
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.grid(True)
    plt.xticks(list(range(0, 100, 2)))
    # sns.swarmplot(y=category_1, x=category_2, hue="Survived", data=train_df)
    plt.legend()
    plt.draw()
    plt.savefig(output_folder+"A_preprocessing/diagrams/2D_"+category_1+"_"+category_2+".png", dpi=300)
    plt.clf()

# dots_2D(category_1="Sex", category_2="Age")



"""
Zaključci:
Preživljavanje je najviše vezano uz: Sex, Age, Pclass.
Također je manje vezano uz Embarked, Fare, Parch, SibSp.
Age i Fare treba transformirati u kategoričku varijablu  i odrediti raspone.

PassangerId, Name i Ticket se izbacuju jer su slučajne varijabla i ne koreliraju s preživljavanjem.
Cabin se izacuje jer za većinu putnika ne postoji podatak.
Izbacuju se nepotpuni podaci jer cjelovitih podataka ima dovoljno za treniranje modela.
"""





###################################        2. Transformiranje podataka        #########################################


# Transformacija kontinuiranih varijabli u kategoričke
def transform_data(input_df, set_for_train):

    # Age se dijeli po osobnim, realnim kategorijama, (5 starosnih kategorija)
    input_df["Age"] = pd.cut(input_df["Age"],bins=[0,5,18,35,60,300],
                            labels=[1,2,3,4,5])

    # Fare se dijeli osvisno o distribuciji na temelju procjene i razlike iz grafa (7 kategorija)
    input_df["Fare"] = pd.cut(input_df["Fare"],bins=[0,10,30,50,80,120,300,600],
                             labels=[1,2,3,4,5,6,7])

    # Sex i Embarked se jednostavno preslikavaju
    input_df["Sex"] = input_df["Sex"].map({"male":0, "female":1})
    input_df["Embarked"] = input_df["Embarked"].map({"C":1, "Q":2, "S":3})

    # Izbacivanje nepotrebnih podataka
    input_df = input_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])       # izbacivanje nepotrebnih kolona

    # Dijeljenje train i test seta
    if set_for_train == True:
        input_df = input_df.dropna()                        # Izbacivanje nepotpunih podataka
        X_data = input_df.drop(columns=["Survived"])        # Za train set treba izbaciti "Survived" kategoriju
        Y_data = input_df["Survived"]

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


# B: dizanje matrice dizajna u prostor više dimenzije (kombinacija pstojećih kategorija)
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
    assert n_components <= len(X_train.columns), "Broj komponenata je prevelik!"  # max komponenata koliko može ostati
    pca = decomposition.PCA(n_components=n_components)
    pca_X_train = pca.fit_transform(scal_std_X_train)
    pca_X_test = pca.fit_transform(scal_std_X_test)

    # Singular value decomposition
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




# Dodatna provjera: Ima li smisla skalirati podatke nakon dizanja u viši prostor značajki - NEMA
def poly_and_scaling():
    poly4_data = polynomial_features_scaling(deg=4)
    poly4_X_train = poly4_data[0]
    poly4_X_test = poly4_data[1]

    scal_data_test = pd.DataFrame()
    categories = [categ for categ in poly4_X_train]
    scal_data_train = pd.DataFrame()
    scaler_model = StandardScaler()

    scal_data_train[categories] = scaler_model.fit_transform(poly4_X_train[categories])
    scal_data_test[categories] = scaler_model.transform(poly4_X_test[categories])

    train_data = scal_data[0]
    test_data = scal_data[1]
    return[train_data, test_data]

poly_and_scaling_data = poly_and_scaling()
poly_scal_train = poly_and_scaling_data[0]
poly_scal_test = poly_and_scaling_data[1]


# Rječnik sa svim test podacima za predviđanje
all_X_test_data = { "X": X_test,
                    "scal_std_X": scal_std_X_test,
                    "scal_MM_X": scal_MM_X_test,
                    "poly3_X": poly3_X_test,
                    "poly4_X": poly4_X_test,
                    "pca_6_X": pca_6_X_test,
                    "pca_5_X": pca_5_X_test,
                    "pca_4_X": pca_4_X_test,
                    "poly_scal_X": poly_scal_test,
                   }

# with open(output_folder+'A_preprocessing/all_X_test_data.pickle', 'wb') as f_test:    # spremanje riječnika u dictdf
#     pickle.dump(all_X_test_data, f_test)


with open(input_data+'all_X_test_data.pickle', 'wb') as f_test:    # spremanje riječnika u dictdf
    pickle.dump(all_X_test_data, f_test)


# Rječnik sa svim ukupnim train podacima
all_X_train_data = {"X": X_train,
                    "scal_std_X": scal_std_X_train,
                    "scal_MM_X": scal_MM_X_train,
                    "poly3_X": poly3_X_train,
                    "poly4_X": poly4_X_train,
                    # "pca_6_X": pca_6_X_train,
                    # "pca_5_X": pca_5_X_train,
                    # "pca_4_X": pca_4_X_train,
                    # "poly_scal_X": poly_scal_train
                    }


# Skup za treniranje se dijeli na skup za treniranje i skup za validaciju modela
X_train_data, X_valid_data, Y_train_data, Y_valid_data, = {}, {}, {}, {}
for data in all_X_train_data:
    # Različiti podaci svaki put
    # X_train, X_valid, y_train, y_valid = train_test_split(all_X_train_data[data], Y_train, random_state=11, test_size=0.1)
    # Uvijek isti podaci:
    X_train, X_valid, y_train, y_valid = train_test_split(all_X_train_data[data], Y_train, test_size=0.1)
    X_train_data[data] = X_train
    X_valid_data[data] = X_valid
    Y_train_data[data] = y_train    # y je ovdje, Y je sve skupa gore
    Y_valid_data[data] = y_valid


# Ukupni podijeljenji podaci
divided_train_data = {
                    "X_train_data": X_train_data,
                    "X_valid_data": X_valid_data,
                    "Y_train_data": Y_train_data,
                    "Y_valid_data": Y_valid_data,
                    }

# Spremanje podataka u rječnik
# with open(output_folder+'A_preprocessing/divided_train_data.pickle', 'wb') as div:
#     pickle.dump(divided_train_data, div)


with open(input_data+'divided_train_data.pickle', 'wb') as f_test:    # spremanje riječnika u dictdf
    pickle.dump(all_X_test_data, f_test)



















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
# a = train_df["Sex"].map(maping)




# tocnost = sklearn.metrics.accuracy_score(y_test, h)

# varijablino ime se ispisuje
# data_name = f'{variable=}'.split('=')[0]



"""
Podaci se mogu i kraće ovako transformirati ali mijenja se zapis u array umjesto DF. Koristiti način u funkciji gore
# Podaci sa standardiziranom transformacijom
scaler_model = StandardScaler()
scal_std_X_train = scaler_model.fit_transform(X_train)
scal_std_X_test = scaler_model.transform(X_test)
# Podaci s MinMax transformacijom
scaler_model = MinMaxScaler()
scal_MM_X_train = scaler_model.fit_transform(X_train)
scal_MM_X_test = scaler_model.transform(X_test)
"""















