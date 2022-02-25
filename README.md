# Titanic_project
Titanic case je školski primjer primjene strojnog učenja za analizu podataka. Input i output podataka se može vidjeti na linku:

Case je podijeljen u 3 dijela:

1. Analiza ulaznih podataka
   - Radi se pregled svih podatka za train i test set: Provjera nedostaju li podaci u pojedinim kategorijama i u kojem postotku, kojim raspodjelama podaci 
     pripadaju te kako se kreće njihov raspon. Nakon toga se crtaju grafovi u odnosu na preživaljavanje. Neporebne kategorije se izbacuju. 
   - Podaci koji nedostaju (većinom Age i Fare) se nadopunjuju, i to uz pomoć kNN algoritma. 
   - Manipulacija odacima: Kako bi se dobila što bolja točnost modela, podaci se skaliraju (standard i MinMax skaliranje), dižu u viši prostor značajki
      te im se reducira broj značajki (pomoću Principal Component Analysis metode) 
   - Svi setovi podataka se slažu u riječnik te se train set dijeli u set za treniranje i validaciju. 
   - Spremanje podataka u pickle format radi lakšeg pristupa kasnije

2. Provjera točnosti modela
   - Za sve modele potrebno je pronaći optimalni set podataka (ulazni set, MM i std skaliranje, podaci dizani u prostor viših značajki i PCA) te prema 
      tome  podešavati model i njegove hiperparametre kako bi se dobila što bolja točnost modela.  
      Modeli koji su analizirani: Logistička regresija, SVM, kNN, Decision Tree, Random Forest i Gaussian Naive Bais.   
   - Za sve modele su nacrtani dijagrami točnosti ovisno o hiperparametrima i korištenim podacima te su podaci spremljeni u excel file. 
   - Napravljena važnost značaki za predikciju.
    
3. Konačni odabir modela
  - Odabrani su optimalni modeli s pripadajućim hiperparametrima te se napravio njihov prosjek predikcija.
  - Zapis predikcija u csv file   
