import pandas as pd
from Read_Data_SingleBlech import Read_Data

## Lade alle Daten der Bleche in ein Dictionary/DataFrame. Diese werden pro Column im Dictionary gespeichert und appended, um so später auf die einzelnen Kräfte zuzugreifen
# Ist nicht für die späteren ML-Modelle, in dieser Funktion werden die Daten für jedes Blech nicht untereinander gespeichert in einem DataFrame sondern nebeneinander, sodass für alle 8 Sensor die 169 Bleche in einer Zeile angeordnet sind
# Wichtig für Konfidenzintervalle etc.
# Bsp:  1-OW-OS Lateral Force   Blech_13    Blech_14    Blech_15    ...
#       0                       62.775466   -114.626576 -80.191106  ...    
#       1                       49.872060   -105.206298 -7.203742   ...    
#       2                       36.256196   -69.979208  -16.624020  ...      
#       3                       104.968812  -62.617142  -63.804572  ... 
#       ...                     ...         ...         ...         ...
#       
#       1-OW-OS Axial Force     Blech_13    Blech_14    Blech_15    ...
#       0                       -5.617249   -18.296184  -6.259221   ...     
#       1                       -16.450516  -1.524682   3.129610    ...     
#       2                       -12.237579  2.688255    3.009241    ...      
#       3                       7.503040    -13.561645  -8.586367   ... 
#       ...                     ...         ...         ...         ...
##############################################################################
def Dataframe_All():
    
    SensorDataFrames = {} # Wird für die erstellung von DataFrames aus den Listen benötigt 

    for i in range(13,170):
        #print(i)
        df, Blech_Nummer = Read_Data(i)
        #print(result)
        # Abfrage wird benötigt aufgrund der Definition von Read_Data. Im Falle das eine Blech Nummer nicht zur Verfügung steht wird None in der Funktion zurückgegeben 
        if df is not None:
            #print(Blech_Nummer)
            #print(df)
            # Gehe jedes Column im DataFrame der Funktion Read Data, also jeden Sensortyp und Axiale und Laterale Kraft durch. Um Später einzeln darauf zuzugreifen
            for Sensor in df.columns:
                #print(Sensor)
                Data = df[Sensor]
                #print(f'Aktuelle Blechnummer {Blech_Nummer} und aktueller Sensor {Sensor}: {Data}') # SO wurde Fehler in Blech 62 gefunden (Abfrage ob alle Bleche richtig geladen werden)
                #print(Data)
                # Erstelle ein leeres DataFrame, insofern noch kein Sensor hinzugefügt wurde
                if Sensor not in SensorDataFrames:
                    SensorDataFrames[Sensor]=[]  #Leere Liste erstellen, insofern noch keine im Dictionary existiert
                    
                # Füge jeden neuen Sensor und neue Kraft der Liste für jeden Sensor hinzu durch append bevor später die Liste zum Dataframe zusammengefügt wird 
                SensorDataFrames[Sensor].append(Data.rename(f'Blech_{Blech_Nummer}'))
    
    # Zusammenfügen zu einem Dataframe der einzelnen Sensoren
    for Sensor, Liste in SensorDataFrames.items():
        SensorDataFrames[Sensor] = pd.concat(Liste, axis=1)

    # Überprüfen der Ergebnisse (Größe der DataFrames)
    for Sensor, df in SensorDataFrames.items():
        print(f"Anzahl der Spalten für Sensor {Sensor}: {df.shape[1]}")
    
    # Gebe das gesamte Dataframe aller CSV zurück
    return SensorDataFrames