# Vorbereitung der Daten für konventionelle Modelle und MLP der Deep Learning Modelle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pandas import DataFrame 
import numpy as np
import random as rnd

# Beim Aufrufen der Funktion müssen die Preprocessed Daten eingegeben werden, die Split Größe, Random_state, den verwendeten Scaler, die batchsize und ob die Daten für den zweiten Test Split gespeichert werden sollen

# Data wird aus der Funktion Load_Data_for_Modelling geladen und im Notebook in diese Funktion für data eingesetzt

# Für standard_split =1 wird der Standard Split durchgeführt
# Für standard_split =2 wird der Blech Split durchgeführt (neue Evaluationsmetrik)

# Für Validation_Data = 1: Werden zusätzlich Validations Daten erstellt (Abhängig von der SIze)

# Zufallsfaktor (seed) für Reproduzierbarkeit definieren

# size: gibt die Größe an Validierungs- und Testdaten an, welche wiederum 50/50 geteilt werden: 0.2 --> 80% Training 10% Validierungs 10% Testdaten

# Scaler: Scaler kann beliebig varriert werden, Standard: StandardScaler

# batchsize muss immer mit angegebn werden, wenn Bereich im Preprocessing variiert wird hier mit beachten

#save =1: Falls Test- und Traingsdaten gespeichert werden sollen

def Split_Scaling(data: DataFrame, size: float=0.2, seed=42, scaler=StandardScaler , Validation_Data =1, standard: int=1, batchsize=1800, save: bool=True, Ordner='.'):
  
  # Hier den Ordner angeben, in welchem die Excel von dem Blech Split gespeichert werden soll
  #Ordner = r'C:\Users\corvi\OneDrive - stud.tu-darmstadt.de\Desktop\Masterthesis\13_ExcelvonDaten_Code'
  
  # Labels definieren
  Columns_drop = ['X_opt-X-Ist','Y_Opt-Y_ist','phi_Opt-phi_ist']
  
  # Standard Train Test Split (Zufällig 80% der Daten werden für das Training verwendet)
  if standard==1:
    # AUfteilen in Features und Labels: X Sind die Features und Y sind die Labels
    X = data.drop(columns = Columns_drop)
    Y = data[Columns_drop]

    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size, random_state=seed)
    
    if Validation_Data ==1:
      
      X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state= seed)
      
    else:
      
      X_val = Y_val = None
  # Blech Split (Extrahieren von einzelnen Blechen vor dem Trainieren)  
  elif standard == 2:
    
    BlechAnzahl = len(data)/batchsize #Anzahl an Bleche in der Gesamtmatrix
    #print(BlechAnzahl)
    Test_Blechmenge = round(BlechAnzahl * size) #Wie viele Bleche für die Testdaten extrahiert werden sollen
    #print(Test_Blechmenge)
    rnd.seed(seed)
    Random_Blech = rnd.sample(range(1, int(BlechAnzahl)),int(Test_Blechmenge)) # Erstelle Random Blech Nummern, welche je nachdem welcher random state vorgegeben wird, für die Testdaten sind
  
    
    Test_data = []
    
    # Umrechnung der Blechnnummern für die genaue Zuordnung, Random_Blech ist lediglich um die richtigen Bleche im Dataframe zu adressieren, da manche Bleche fehlerhaft sind und rausgelassen wurden (insgesamt 142 Bleche)
    # Hiermit werden die Nummern zur Adressierung im Dataframe umgerechnet in die wahren Blechnummern für eine genaue Zuordnung 
    angepasste_Blechnummern = []
    for x in Random_Blech:
        new_x = x + 13
        if new_x >= 86:
            new_x += 12
        if new_x >= 121:
            new_x +=1
        if new_x >=139:
            new_x +=1
        if new_x >=157:
            new_x +=1 
        angepasste_Blechnummern.append(new_x)
    
    # Aufteilung der rausgezogenen Bleche in Test und Validation Data für die Hyperparametervalidierung 
    if Validation_Data == 1:
      single_val_list =[]
      single_test_list = []
      # Aufteilung der Bleche in Validation and 
      Random_Blech_val = Random_Blech[:len(Random_Blech)//2]
      Random_Blech_test = Random_Blech[len(Random_Blech)//2:]
      print(f'Blech Nummern der Validations Daten: {angepasste_Blechnummern[:len(angepasste_Blechnummern)//2]}')
      print(f'Blech Nummern der Test Daten: {angepasste_Blechnummern[len(angepasste_Blechnummern)//2:]}')
      
       # Iteriere über die Blechnummern, um die Daten für die Validationsdaten zu extrahieren
      for i in Random_Blech_val:
          single_val = data.iloc[batchsize*i:batchsize*(i+1)]
          #print(single_Features)
          single_val_list.append(single_val)
          
        # Iteriere über die Blechnummern, um die Daten für die Testdaten zu extrahieren
      for i in Random_Blech_test:
          single_test = data.iloc[batchsize*i:batchsize*(i+1)]
          #print(single_Features)
          single_test_list.append(single_test)
      
      # Füge die einzelnen Bleche aus der Liste zusammen
      df_test = pd.concat(single_test_list, axis=0)
      df_val = pd.concat(single_val_list, axis=0)
      # Validations und Testdaten aus den Trainingsdaten  anhand der Indizes
      df_train = data.drop(df_test.index.append(df_val.index))
      
      # Random Sample der Dateien
      df_train= df_train.sample(frac=1, random_state=seed)
      df_test = df_test.sample(frac=1, random_state=seed)
      df_val = df_val.sample(frac=1, random_state=seed)
      
      #Aufteilung der Trainings und Testdaten in Features und Labels 
      X_train, X_test, X_val = df_train.drop(columns = Columns_drop), df_test.drop(columns=Columns_drop), df_val.drop(columns=Columns_drop)
      Y_train, Y_test, Y_val = df_train[Columns_drop], df_test[Columns_drop], df_val[Columns_drop]
      
      # Falls keine Validationsdaten benötigt werden, iteriere lediglich über die oben definierten Blechnummern
    else:
      
      # Iteriere über die Blechnummern, um die Daten für den Testsplit zu extrahieren
      for i in Random_Blech:
        single = data.iloc[batchsize*i:batchsize*(i+1)]
        #print(single)
        Test_data.append(single)
      
      # Dataframes für Train und Test Daten
      df_test = pd.concat(Test_data, axis=0)
      # Löschen der Testdaten aus den Trainingsdaten
      df_train = data.drop(df_test.index)
      
      # Random Sample der Dateien
      df_train= df_train.sample(frac=1, random_state=seed)
      df_test = df_test.sample(frac=1, random_state=seed)

      #Aufteilung der Trainings und Testdaten in Features und Labels 
      X_train, X_test = df_train.drop(columns = Columns_drop), df_test.drop(columns=Columns_drop)
      Y_train, Y_test = df_train[Columns_drop], df_test[Columns_drop]
    
    #Falls die extrahierten Testdaten und Trainingsdaten in Excel angeschaut werden möchten
    if save:
      for Column1,Column2 in zip(df_test.columns,df_train.columns):
          df_test[Column1] = df_test[Column1].astype(str).str.replace('.', ',')
          df_train[Column2] =df_train[Column2].astype(str).str.replace('.', ',')
          
      #df_test.to_csv(f'{Ordner}\Testdaten_BlechSplit.csv', index=True, sep=';')
      #df_train.to_csv(f'{Ordner}\Trainingsdaten_BlechSplit.csv', index=True, sep=';')
  
  else:
    print('Daten können nicht eingelesen werden. Für standard_split 1 angeben, um Standard Split durchzuführen. Bei 2 wird ein Split nach Blechen durchgeführt')
    return None
    
  #print(len(X_test))
  #print(len(Y_test))
  
  # Normalisierung oder Skalierung nur auf den Trainingsdaten anwenden
  # Es werden für die Kraftdaten andere Scaler eingesetzt als für die Positionsdaten aufgrund der unterschiedlichen Einheiten und Größen. Gleiches gilt für Phi, sowie x und y
  scaler_X_position = scaler()
  scaler_X_phi = scaler()
  scaler_X_forces = scaler()

  # Skalierung der X (Features) seperat für die Kräfte und Positionen, welche dann wieder in einem Dataframe zusammengefügt werden
  for scaler, columns in zip([scaler_X_forces, scaler_X_position, scaler_X_phi], [data.columns[:8], data.columns[8:10], ['phi-Ist']]):
    X_train_scaled_columns = scaler.fit_transform(X_train[columns])
    X_test_scaled_columns = scaler.transform(X_test[columns])
    
    # Ersetzen mit den skalierten Werten
    X_train[columns] = X_train_scaled_columns
    X_test[columns] = X_test_scaled_columns
    
    if Validation_Data ==1:
      X_val_scaled_columns = scaler.transform(X_val[columns])
      X_val[columns] = X_val_scaled_columns
  
  # Je nachdem ob Validationsdaten benötigt werden erfolgt die entprechende Rückgabe der Daten
  if Validation_Data ==1:
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
  
  else:
    return X_train, X_test, Y_train, Y_test