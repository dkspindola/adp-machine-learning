# Dient zum Laden der Daten für den Fall einer Feature Extraktion aus der Preprocessing Funktion

# Datei zum Zusammenfügen der Labels mit den gesamten Blechen zum Trainieren der Modelle
# Vorverarbeitung der Daten der einzelnen Bleche findet in einer anderen Datei statt (Bereich definieren, Offset eliminieren etc, Einheitliche Länge)
# Berechnung dauert aktuell etwa 2min 43s, wird aber noch kürzer durch Identifizieren der Bereiche --> Deutlich weniger Datenpunkte
##### Achtung Bleche 86-97 sind fehlerhaft und müssen noch Berücksichtigt werden ############


from Read_Data_SingleBlech import Read_Data
from PreProcessing_SingleBlech_Function_Features import Preprocessing
from Get_Labels import Get_Label
import pandas as pd

# n gibt den Startpunkt an und m den Endpunkt bei den Features spielt dieser Wert eine untergeordnete ROlle

def Data_for_Model(n,m):

    # Lädt die Labels aus der Excel in der Funktion GetLabels richtig rein
    Labels = Get_Label()
    #Leere liste die mit den Daten in der Schleife appendet werden kann 
    Dataframe_Label = []
    numbers = list(range(13, 86)) + list(range(98, 170)) #Um die fehlerhaften Messungen von Blech 86-97 rauszuwerfen 

    #print(Labels)

    for i in numbers:
        
        Daten_Blech, Features_df, Blech_Nummer = Preprocessing(i,n,m,0)
        
        if Daten_Blech is not None:
            
            Labels_Blech = Labels[Labels['BlechNr'] == Blech_Nummer]                       #Vergleich der gezogenen BlechNummer aus der ReadData Datei und der BlechNr aus den Labels
            #print(Labels_Blech)
            print(i)
            Labels_Erweitern = pd.concat([Labels_Blech]*len(Features_df), ignore_index=True)    #Erweiterung der Labels auf die Länge des Dataframes der Rohdaten des Bleches
            #print(Labels_Erweitern)
            Data = pd.concat([Features_df.reset_index(drop=True),Labels_Erweitern.reset_index(drop=True)], axis =1) #   Zusammenführen der Labels mit den Kraftdaten zu einem Dataframe
            print(Data.shape)
            print(Data)
            Dataframe_Label.append(Data)  #Alle Dataframes appenden um am Ende einen Dataframe mit den Kraftdaten und Labels untereinander hat 
            print(len(Dataframe_Label))
            
    df = pd.concat(Dataframe_Label, ignore_index=True, axis=0) #axis =0 ermöglicht das Zusammenführen untereinander
    print(len(df))

    print(print(f"Anzahl der Zeilen im finalen DataFrame: {df.shape[0]}"))
    #print(df.columns)
    df = df.drop(columns=['Gerade/NichtGerade','BlechNr']) # Droppen der für das Modellieren nicht benötigten Zeilen 
    #print(df)
    
    return df