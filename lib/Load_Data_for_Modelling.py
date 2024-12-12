from lib.Read_Data_SingleBlech import Read_Data
from lib.PreProcessing_SingleBlech_Function import Preprocessing
from lib.Get_Labels import Get_Label
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# n ist der Startpunkte in dem extrahierten stationären Datenbereich 
# m gibt den Endpunkte an im stationären Datenbereich bevor das Blech aus dem vorletzten Stich herausläuft
# Standard für gesamten Bereich: [0,1800]

def Get_data(n=0, m=1800, Interpolation=0, Anzahl_Interpolationen=1):
    
    # Lädt die Labels aus der Excel in der Funktion GetLabels richtig rein
    Labels = Get_Label()
    
    #Leere liste die mit den Daten in der Schleife appendet werden kann 
    Dataframe_Label = []
    
    numbers = list(range(13, 86)) + list(range(98, 170)) #Um die fehlerhaften Messungen von Blech 86-97 rauszuwerfen, kann für neue Daten angepasst werden, wenn diese die gleiche Bezeichnung im Ordner haben (siehe Read_Data_SingleBlech)

    #Iterieren über die Bleche und Preprocessing jedes einzelnen Bleches mit der Funktion Preprocessing 

    for i in numbers:
        
        # Reinladen der vorverarbeiteten Daten (Zentriertes_df ist nur zentriert und EInlauf abgeschnitten)
        # Daten Blech ist der stationäre Datenbereich
        Daten_Blech, Zentriertes_df, Blech_Nummer = Preprocessing(i,n,m,0)
        
        if Daten_Blech is not None: #IF Abfrage wird benötigt falls einzelne Blechdaten (121, 139, 157) fehlerhaft sind dann wird in der Read Data Funktion None zurückgegeben
            
            Labels_Blech = Labels[Labels['BlechNr'] == Blech_Nummer]                       #Vergleich der gezogenen BlechNummer aus der ReadData Datei und der BlechNr aus den Labels
            #print(Labels_Blech)
            Labels_Erweitern = pd.concat([Labels_Blech]*len(Daten_Blech), ignore_index=True)    #Erweiterung der Labels auf die Länge des Dataframes der Rohdaten des Bleches
            #print(Labels_Erweitern)
            Data = pd.concat([Daten_Blech.reset_index(drop=True),Labels_Erweitern.reset_index(drop=True)], axis =1) #   Zusammenführen der Labels mit den Kraftdaten zu einem Dataframe
            #print(Data)
            Dataframe_Label.append(Data)  #Alle Dataframes der einzelnen Bleche appenden um am Ende einen Dataframe mit den Kraftdaten und Labels untereinander hat 
            
    # Zusammenfügen der Liste aus Kraftisgnalen und Positionen der Bleche zu einem Dataframe        
    df = pd.concat(Dataframe_Label, ignore_index=True, axis=0) #axis =0 ermöglicht das Zusammenführen untereinander

    # Falls interpolierte Daten erzeugt werden sollen
    if Interpolation == 1:
        
        # Gibt die verschiedenen Fehleinstellungen der Rollformanlage an, sodass die Labels auch richtig erstellt werden und nicht unterschiedliche Positionen für ein gerades Blech haben
        Bleche_Einstellung = [range(13,23), range(24,32), range(33,41), range(42,51), range(52,60), range(61,77), range(78,105), range(106,122), range(123,139), range(140,154), range(155,169)]
        
        # Anzahl_Interpolationen 
        Interpolationsparam = np.linspace(0, 1, Anzahl_Interpolationen + 2)[1:-1]  # Erzeugt die entsprechenden Parameter für die Interpolation
        print(Interpolationsparam)
        
        # Neue DataFrames für die interpolierten Werte erstellen
        interpolated_force = []
        Mittelwert_labels = []
        Mittelwert_Positionen = []
        Liste_new = []
        
        # Iterieren über die einzelnen Versuchsreihen
        for Blech_range in Bleche_Einstellung:
            
            for i in Blech_range:
            
                # Listen für die verschiedenen Teile des finalen DataFrames
                interpolated_force = []
                Mittelwert_labels = []
                Mittelwert_Positionen = []
                
                # Auswahl der Bleche zum Vergleich um dazwischen neue zu erstellen
                df_1 = df[df['BlechNr'] ==i].reset_index(drop=True)
                #print(df_1)
                df_2 = df[df['BlechNr'] == i+1].reset_index(drop=True)
                #print(df_2)
                
                # Überprüfen, ob df_1 und df_2 nicht leer sind
                if df_1.empty or df_2.empty:
                    continue  # Überspringt die Iteration, wenn eine der DataFrames leer ist
                
                #if df_1['X-Ist'] != df_2['X-Ist'] or df_1['Y-Ist'] != df_2['Y-Ist'] or df_1['phi-Ist'] != df_2['phi-Ist']: Funktioniert nicht
                
                # Falls Dataframes nicht leer (Verschiedene Bleche wurden ausgelassen weil diese fehlerhaft sind)
                if not (df_1['X-Ist'].equals(df_2['X-Ist']) and df_1['Y-Ist'].equals(df_2['Y-Ist']) and df_1['phi-Ist'].equals(df_2['phi-Ist'])):
                    
                    # Falls mehrere Zwischenschritte inerpoliert werden wird hier über diese iteriert 
                    for t in Interpolationsparam:
                        
                        interpolated_force = []
                        Mittelwert_labels = []
                        Mittelwert_Positionen = []
                    
                        # Neue Ist Positionen zwischen zwei Blechen bestimmen 
                        for columns in df.columns[9:12]:
                            
                            #Auswahl der Columns der Ist Positionen
                            Position_1 = df_1[columns]
                            Position_2 = df_2[columns]
                            
                            New_Position = Position_1 * (1 - t) + Position_2 * t  # Interpolation zwischen den Blechen
                            #print(New_Position)
                            Mittelwert_Positionen.append(New_Position)
                            #print(Mittelwert_df_pos)
                            
                        # Neue Labels zwischen zwei Blechen bestimmen 
                        for columns_label in df.columns[13:16]:
                            #print(columns_label)
                            # Auswahl der Columns der Labels
                            Label_1 = df_1[columns_label]
                            Label_2 = df_2[columns_label]
                            
                            New_Labels = Label_1 * (1 - t) + Label_2 * t  # Interpolation
                            #print(New_Labels)
                            # Mittelwert_df_label = pd.DataFrame(New_Labels, columns=[f'Mittelwert_{columns_label}'])
                            Mittelwert_labels.append(New_Labels)
                        
                        # Neue Kraftsignale zwischen zwei Blechen bestimmen
                        for columns in df.columns[:8]:
                            
                            # Auswahl der COlumns der Kräfte
                            Kraft_Blech1 = df_1[columns]
                            Kraft_Blech2 = df_2[columns]
                            
                            # Falls notwendig: Interpolationsmethode auswählen (z.B. linear)
                            interp_func = interp1d([0, 1], [Kraft_Blech1, Kraft_Blech2], axis=0, kind='linear')
                            
                            interpolated_y = interp_func(t)
                            
                            # Erstelle ein neues DataFrame mit den interpolierten Werten und füge es zu den interpolated_dfs hinzu
                            interpolated_df = pd.DataFrame(interpolated_y, columns=[f'{columns}'])
                            interpolated_force.append(interpolated_df)
                            #print(interpolated_force)
                        
                        # Zusammenführen der interpolierten DataFrames zu einem Gesamtdatensatz
                        New_forces = pd.concat(interpolated_force, axis=1)
                        New_Position = pd.concat(Mittelwert_Positionen, axis=1)
                        New_Labels = pd.concat(Mittelwert_labels, axis=1)

                        #print(New_Position)
                        #print(New_Labels)
                        # Zusammenführen der Features und Labels
                        New_df = pd.concat([New_forces, New_Position, New_Labels], axis=1)
                        
                        # In Liste appenden für alle Zwischenräume zwischen Blechen
                        Liste_new.append(New_df)
        
        # Zusammenfügen aller neuen Bleche
        final_df = pd.concat(Liste_new, axis=0)  
        Interpoliertes_df = final_df.reset_index(drop=True)
        print(Interpoliertes_df)

        new_df = df.drop(columns=['Gerade/NichtGerade','BlechNr']) # Droppen der für das Modellieren nicht benötigten Zeilen aus den Versuchsdaten
        #print(new_df)
        # Zusammenfügen von neuen interpolierten Daten und ursprünglichen Versuchsdaten
        Data_for_Model = pd.concat([new_df, Interpoliertes_df], axis =0)
        Data_for_Model = Data_for_Model.reset_index(drop=True)
        
        print(print(f"Anzahl der Zeilen im finalen DataFrame der gesamten Daten mit interpolierten Daten: {Data_for_Model.shape}"))
        print(print(f"Anzahl der Zeilen im finalen DataFrame nur interpolierte Daten ohne die Versuchsdaten: {Interpoliertes_df.shape}"))
        
        #print(Data_for_Model)
        
    else:
        
        #print(df.columns)
        df_ohneInt = df.drop(columns=['Gerade/NichtGerade','BlechNr']) # Droppen der für das Modellieren nicht benötigten Zeilen 
        print(print(f"Anzahl der Zeilen im finalen DataFrame: {df_ohneInt.shape}"))
    
    # Wenn Interpolierte Daten geladen werden, wird das Gesamtdataframe aus interpolierten und Versuchsdaten (Data_for_Model) und nur das interpolierte Dataframe zurückgegeben
    if Interpolation == 1:
    
        return Data_for_Model, Interpoliertes_df
    
    # Ohne Interpolation nur das normale Dataframe der Versuchsdaten aus Feature und Labels
    else:
        return df_ohneInt