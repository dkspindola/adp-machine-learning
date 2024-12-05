# Datei zum Reinladen der Daten aus den Versuchen, dabei wird zunächst jedes Blech einzeln geladen
import glob
import os
import pandas as pd


def Read_Data(Blech_Nummer, NewGeo=0):
    
    # Hier den Pfad angeben, in dem die CSV Dateien aus dem Versuch abliegen
    Ordner_Pfad= r"C:\Users\corvi\OneDrive - stud.tu-darmstadt.de\Desktop\Masterthesis\10_Daten_Versuche"
    
    # Falls die neue Geometrie reingeladen werden soll (kurzes Blech)
    if NewGeo==1:
        Pfad= os.path.join(Ordner_Pfad, f'newgoem{Blech_Nummer}_*/')
        
    # Für die ursprünglichen Daten (DC04 Blech 2m) 
    else:
        Pfad= os.path.join(Ordner_Pfad, f'Blech{Blech_Nummer}_*/')
        
    #Zusammenfügen des Pfades und Initialisierung von Data_merged um Fehler bei nicht vorhandenen Blechen zu umgehen
    KorrekterPfad = glob.glob(Pfad)
    EinzelneCSV=[]
    Data_merged = None
    
    #print(KorrekterPfad)
    
    # Falls eine Blech Nummer/Ordner nicht existiert wird am Ende None zurückgegeben --> Muss beim Aufrufen der Funktion berücksichtigt werden
    if KorrekterPfad:
        # Iteration über alle Pfade in dem Ordner
        for i in KorrekterPfad:
            CSV_file = glob.glob(os.path.join(i,'*.csv'))
            #print(CSV_file)
            for n in CSV_file:
                # Überspringen der merged CSV Datei und Reinladen der Dateien jedes Sensors (Merged teilweise fehlerhaft und oft unterschiedlich)
                if 'merged' not in n:
                    
                    #Reinladen der CSV
                    df = pd.read_csv(n, sep=',', header=16, index_col=False, on_bad_lines='skip') 
                    # Skip Bad lines wird benötigt weil die letzte Zeil in jeder CSV anders formatiert ist als die anderen und mehr Einträge aufgrund der sep =, besitzt 
                    # Es wird bei Zeile 16 gestartet, da die vorherigen Zeilen irrelevant sind
                    
                    # Falls im Versuch doch die anderen Kräfte (Mittelwert etc.) nicht abgewählt wurden, werden diese hier gelöscht 
                    if len(df.columns) >= 3:
                        df = df.iloc[:, :3]
                    # Aus dem Namen den aktuellen Sensor ziehen, um diesen mit in die Columns zu packen    
                    file_name= os.path.basename(n)
                    Sensor= file_name.split('_')[0]
                    
                    #print(Sensor)
                    #print(df)

                    #Columns = df.iloc[0] Hat nicht funktioniert, da er immer eine 0 in die Zeile der Columns gepackt hat
                    Columns = ['Time in ms', 'Lateral Force', 'Axial Force'] # Daher Händisch, alle CSV sind in dieser Reihenfolge aufgebaut
                    
                    df.columns = [f'{Sensor} {col}' for col in Columns]
                    
                    # Alle Spalten mit der Zeit in ms löschen, da Datenpunkte (Index) verwendet wird
                    df = df.drop(columns=[f'{Sensor} Time in ms'])
            
                    # Umwandeln des Datentypes der Einträge von Strings in floats
                    df = df.astype(float)
                    
                    # Alle einzelnen Sensoren werden in die Liste aufgenommen
                    EinzelneCSV.append(df)
                    #print(df)
        # Wenn Dataframe nicht None wie bei Blechen die nicht vorhanden sind wird die Liste zusammengefügt      
        if EinzelneCSV:
            Data_merged = pd.concat(EinzelneCSV, axis=1)            
        #print(EinzelneCSV)
    # Falls das Dataframe nicht None ist wird das Dataframe und die Blechnummer zurück gegeben, ansonsten nur die Blechnummer
    if Data_merged is not None:
        return Data_merged, Blech_Nummer
    else:
        return None, Blech_Nummer
