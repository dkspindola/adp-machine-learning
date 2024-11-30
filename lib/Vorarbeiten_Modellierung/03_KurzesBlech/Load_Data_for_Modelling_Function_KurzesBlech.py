# Für das Kurze Blech
# Datei zum Zusammenfügen der Labels mit den gesamten Blechen zum Trainieren der Modelle 

from Read_Data_SingleBlech import Read_Data
from PreProcessing_SingleBlech_Function_KurzesBlech import Preprocessing
from Get_Labels_KurzesBlech import Get_Label
import pandas as pd

# n und m gibt die Anzahl an Datenpunkte im stationären Datenbereich an, die man pro Kraftverlauf pro Blech haben möchte: Standard: [0,1400] für das kurze Blech

def Data_for_Model(n,m):

    # Lädt die Labels aus der Excel in der Funktion GetLabels richtig rein
    Labels = Get_Label()
    #Leere liste die mit den Daten in der Schleife appendet werden kann 
    Dataframe_Label = []
    numbers = (3,5,6) #Um die fehlerhaften Messungen der anderen Bleche rauszuwerfen nur 3,5 und 6 sind fehlerfrei

    #Iterieren über die Bleche und Preprocessing jedes einzelnen Bleches mit der Funktion Preprocessing 
    for i in numbers:
        
        Daten_Blech, Zentriertes_df, Blech_Nummer = Preprocessing(i,n,m,0)
        
        if Daten_Blech is not None: #IF Abfrage wird benötigt falls einzelne Blechdaten fehlerhaft sind dann wird in der Read Data Funktion None zurückgegeben
            
            Labels_Blech = Labels[Labels['BlechNr'] == Blech_Nummer]                       #Vergleich der gezogenen BlechNummer aus der ReadData Datei und der BlechNr aus den Labels
            #print(Labels_Blech)
            Labels_Erweitern = pd.concat([Labels_Blech]*len(Daten_Blech), ignore_index=True)    #Erweiterung der Labels auf die Länge des Dataframes der Rohdaten des Bleches
            #print(Labels_Erweitern)
            Data = pd.concat([Daten_Blech.reset_index(drop=True),Labels_Erweitern.reset_index(drop=True)], axis =1) #   Zusammenführen der Labels mit den Kraftdaten zu einem Dataframe
            #print(Data)
            Dataframe_Label.append(Data)  #Alle Dataframes appenden um am Ende einen Dataframe mit den Kraftdaten und Labels untereinander hat 
    
    # Zusammenführen der einzelnen Dataframes pro Blech      
    df = pd.concat(Dataframe_Label, ignore_index=True, axis=0)

    # Finales Dataframe
    print(print(f"Anzahl der Zeilen im finalen DataFrame: {df.shape[0]}"))
    #print(df.columns)
    df = df.drop(columns=['Gerade/NichtGerade','BlechNr']) # Droppen der für das Modellieren nicht benötigten Zeilen aus der Labels Datei 
    #print(df)
    return df