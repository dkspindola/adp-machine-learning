from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np

# Funktion für die Aufteilung der Daten mit den extrahierten Features, dazu wird die Datengröße auf 1 gesetzt und lediglich die Bleche zufällig durchmischt
# Ein train test split 2 ist somit nicht notwendig bei dieser Funktion 
# Funktionsweise analog zur Splitting_Scaling_Function Funktion für alle Daten ohne Feature Extraktion

def Split_Scaling(data, size=0.2, random=42, Scaler=StandardScaler , Train_Test_Split=1, Datengröße=1, Speichern=0):
  
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.preprocessing import MinMaxScaler
  import pandas as pd 
  import numpy as np
  import random as rnd
  
  Ordner = r'C:\Users\corvi\OneDrive - stud.tu-darmstadt.de\Desktop\Masterthesis\13_ExcelvonDaten_Code'
  
  Columns_drop = ['X_opt-X-Ist','Y_Opt-Y_ist','phi_Opt-phi_ist']
  
  if Train_Test_Split == 1:
    # AUfteilen in Features und Labels X Sind die Features und Y sind die Labels
    X = data.drop(columns = Columns_drop)
    Y = data[Columns_drop]

    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= size, random_state= random)
    
  elif Train_Test_Split == 2:
    
    #BlechAnzahl = len(data)/Datengröße
    #print(BlechAnzahl)
    #Test_Blechmenge = round(BlechAnzahl * size)
    #print(Test_Blechmenge)
    rnd.seed(random)
    #Random_Blech = rnd.sample(range(1, int(BlechAnzahl)),int(Test_Blechmenge))
    Random_Blech =rnd.sample(range(142),int(142*size))
    print(Random_Blech)
    Test_data = []
    for i in Random_Blech:
      single = data.iloc[Datengröße*i:Datengröße*(i+1)]
      #print(single)
      Test_data.append(single)
    
      df_test = pd.concat(Test_data, axis=0)
      df_train = data.drop(df_test.index)
    
    df_train= df_train.sample(frac=1, random_state=random)
    df_test = df_test.sample(frac=1, random_state=random)

    X_train, X_test = df_train.drop(columns = Columns_drop), df_test.drop(columns=Columns_drop)
    Y_train, Y_test = df_train[Columns_drop], df_test[Columns_drop]
    
    if Speichern ==1:
      for Column1,Column2 in zip(df_test.columns,df_train.columns):
          df_test[Column1] = df_test[Column1].astype(str).str.replace('.', ',')
          df_train[Column2] =df_train[Column2].astype(str).str.replace('.', ',')
          
      df_test.to_csv(f'{Ordner}\Testdaten_BlechSplit.csv', index=True, sep=';')
      df_train.to_csv(f'{Ordner}\Trainingsdaten_BlechSplit.csv', index=True, sep=';')
  
  else:
    print('Daten können nicht eingelesen werden. Für Train_Test_Split 1 angeben, um Standard Split durchzuführen. Bei 2 wird ein Split nach Blechen durchgeführt')
    return None
  
    
  print(len(X_test))
  print(len(Y_test))
  print(Scaler)
  # Normalisierung oder Skalierung nur auf den Trainingsdaten anwenden
  # Es werden für die Kraftdaten andere Scaler eingesetzt als für die Positionsdaten aufgrund der unterschiedlichen Einheiten und Größen. Gleiches gilt für Phi, sowie x und y
  scaler_X_position = Scaler()
  scaler_X_phi = Scaler()
  scaler_X_forces = Scaler()

  # Z-Score der X (Features) seperat für die Kräfte und Positionen, welche dann wieder in einem Dataframe zusammengefügt werden
  for scaler, columns in zip([scaler_X_forces, scaler_X_position, scaler_X_phi], [data.columns[:8], data.columns[8:10], ['phi-Ist']]):
    X_train_scaled_columns = scaler.fit_transform(X_train[columns])
    X_test_scaled_columns = scaler.transform(X_test[columns])
    
    X_train[columns] = X_train_scaled_columns
    X_test[columns] = X_test_scaled_columns
    

  return X_train, X_test, Y_train, Y_test