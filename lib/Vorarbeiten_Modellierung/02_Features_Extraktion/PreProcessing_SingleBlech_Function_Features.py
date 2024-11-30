# Dient zur Vorbereitung der Rohdaten auf Features indem 5 Features dafür extrahiert werden
from Read_Data_SingleBlech import Read_Data
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from tsfresh import extract_features

# Einzelne BlechNummern können übergeben werden und werden in diese Funktion vorverarbeitet (Zentriert, Offset entfernt etc.)
#Anpassungen können hier für das Preprocessing erfolgen
# Die Funktion wird wiederum in der Funktion Load_Data_for_Modelling_Function aufgerufen, um alle aufgenommenen Bleche zu laden und in einem Dataframe für das Trainieren der Modelle zu speichern
# Die Funktion Preprocessing greift auf die Funktion Read_Data zu, welche lediglich die Daten aus dem Ordner für jedes einzelne Blech rein lädt. Falls eine Datei nicht exisitert wird None zurückgegeben

# Die Sprünge beim Einlaufen aller Kraftverläufe werden automatisch nicht berücksichtigt 
def Preprocessing(BlechNummer, n, m, Speichern):
    
    df, BlechNummer = Read_Data(BlechNummer)
    
    # If-Abfrage wird benötigt weil in der Funktion Read_Data None retruned wird, falls eine Blech Nummer nicht existiert 
    if df is not None:
        #print(df)
        Index_List =[]
        df_liste =[]

        # Kraftsignal Zentrieren durch Mittelwertbildung bei der Schwankung um 0
        def Zentrieren(Signal):
            for col in Signal.columns:
                
                Mittelwert = Signal[col].iloc[:200].mean()
                #print(Mittelwert)
                
                Signal[col]=Signal[col] - Mittelwert
                
            return Signal

        Zentriertes_df = Zentrieren(df) 
        #print(Zentriertes_df)
        
        # Abschneiden des Einlaufs und Festlegen auf n Datenpunkte
        for col in Zentriertes_df.columns:
            
            #max_index = df[col].iloc[:2000].abs().idxmax()  ### Mit Max Wert nicht sinnvoll beispiel 2-OW-RS Lateral von Blech 70
            count = 0
            Index = None

            for index, value in Zentriertes_df[col].items():
                if value >150 or value <-150:
                    count +=1
                
                if count >=10:
                    Index = index
                    break
            
            if Index is not None:
                Index_List.append((col, Index))
                
        for col, index in Index_List:
            df1 = Zentriertes_df[col].iloc[index+300:index+8000]
            df1 =df1.reset_index(drop=True)
            #print(df1)
            df_liste.append(df1)
            
        Preprocessed_Data = pd.concat(df_liste, axis=1)
        Preprocessed_Data = Preprocessed_Data.reset_index(drop=True)
        Data_seg = Preprocessed_Data.iloc[3400:5200]
        Data_seg = Data_seg.reset_index(drop=True)
        Data_seg = Data_seg.iloc[n:m]

        # Erstellung der Features
        columns_f = Data_seg.columns.tolist()
        Data_seg['id'] = 1
        Data_seg['time'] = range(m)
        Parameter = {
            "mean": None,
            "median": None,
            'maximum':None,
            'minimum':None,
            'standard_deviation': None
        }
        # DataFrame von Wide-Format in Long-Format umwandeln
        #print(Data_seg)
        Data_long = pd.melt(Data_seg, id_vars=['id', 'time'], var_name='kind', value_name='value')
        #print(Data_long)

        #print(type(Data_seg.columns))
        # Features extrahieren 
        Feature_df = extract_features(Data_long, column_id='id', column_sort = 'time', column_kind= 'kind', column_value= 'value', disable_progressbar=True, default_fc_parameters=Parameter)
        print(Feature_df)
        print(Feature_df.dtypes)
        
        return Data_seg, Feature_df, BlechNummer
    
    else:
        return None, None, BlechNummer
    
