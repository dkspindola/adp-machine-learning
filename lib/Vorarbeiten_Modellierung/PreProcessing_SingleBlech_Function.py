from Read_Data_SingleBlech import Read_Data
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Einzelne BlechNummern können übergeben werden und werden in diese Funktion vorverarbeitet (Zentriert, Offset entfernt etc.)
# Falls die Plots gespeichert werden sollen muss bei der Funktion für Speichern eine 1 übergeben werden
# Anpassungen können hier für das Preprocessing erfolgen
# Die Funktion wird wiederum in der Funktion Load_Data_for_Modelling_Function aufgerufen, um alle aufgenommenen Bleche zu laden und in einem Dataframe für das Trainieren der Modelle zu speichern
# Die Funktion Preprocessing greift auf die Funktion Read_Data_SingleBlech zu, welche lediglich die Daten aus dem Ordner für jedes einzelne Blech rein lädt. Falls eine Blech Datei nicht exisitert wird None zurückgegeben

# n gibt die Anzahl an Datenpunkte an, die man pro Kraftverlauf pro Blech haben möchte: Der stationäre Bereich liegt im Wertebereich für x von [0,1800] und stellt die Maximalwerte da, da dies den Bereich darstellt in dem das Blech gerichtet wird. 
# Die Sprünge beim Einlaufen aller Kraftverläufe werden automatisch nicht berücksichtigt 
def Preprocessing(BlechNummer, n, m, Speichern):
    
    # Reinladen der einzelnen Kraftsignale der Bleche
    df, BlechNummer = Read_Data(BlechNummer)
    
    # If-Abfrage wird benötigt weil in der Funktion Read_Data None retruned wird, falls eine Blech Nummer nicht existiert (Manche Versuche waren fehlerhaft)
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
        
        # Abschneiden des Einlaufs (Zeitlichen Offset entfernen) Festlegen auf n Datenpunkte
        for col in Zentriertes_df.columns:
            
            count = 0
            Index = None
            # Wenn mehr als 10 mal der Wert + oder - 150 überschritten wird, wird der neue Startpunkt der Daten festgelegt
            for index, value in Zentriertes_df[col].items():
                if value >150 or value <-150:
                    count +=1
                
                if count >=10:
                    Index = index
                    break
            
            if Index is not None:
                Index_List.append((col, Index))
        
        # Abschneiden der ersten Peaks, um stationären Bereich für alle Bleche zu identifizieren (Nach dem festlegen des Starpunkte oben, werden hier aktuell die ersten 300 Werte abgeschnitten)
        for col, index in Index_List:
            df1 = Zentriertes_df[col].iloc[index+300:index+8000] # Für stationären Bereich 
            df1 =df1.reset_index(drop=True)
            #print(df1)
            df_liste.append(df1)
        
        # Preprocessed Data stellt den Bereich aller Bleche nach zeitlichem Zentireren und abschneiden der ersten 300 Werte dar. --> Identifikation stationärer Bereich
        Preprocessed_Data = pd.concat(df_liste, axis=1)
        Preprocessed_Data = Preprocessed_Data.reset_index(drop=True)
        
        # Data_zentriert schneidet die Werte nach verlassen des 5. (vorletzten Stiches) ab, da diese nicht relevant sind, behält aber den EInlauf bis zum stationären Bereich bei
        Data_zentriert = Preprocessed_Data.iloc[0:5200]
        
        # Data_stat ist der für die Modellierung festgelegte stationäre Bereich 
        Data_stat = Preprocessed_Data.iloc[3400:5200]
        Data_stat = Data_stat.reset_index(drop=True)
        # Hier kann die Anzahl an Daten durch Übergabe von n und m beliebig angepasst werden
        Data_stat = Data_stat.iloc[n:m]
    

        ########################################   PLOTS  #####################################################
        # Falls Speichern 1 werden für die ausgewählten Blech Nummern die Plots erstellt
        if Speichern ==1:
            
            ### Hier den entsprechenden Sicherungsordner für die Plots angeben ###
            Sicherungsordner = f'C:\\Users\\corvi\\OneDrive - stud.tu-darmstadt.de\\Desktop\\Masterthesis\\15_Plots\\Plots_{BlechNummer}'
            #print(Sicherungsordner)
            # Falls Sicherungsordner noch nicht existiert, wird dieser erstellt
            if os.path.exists(Sicherungsordner):
                print(f'{Sicherungsordner} --> Existiert bereits')
            else:
                os.makedirs(Sicherungsordner)
                
            ########################################   PLOT Zentrierung der Kräfte mit Einlauf  #####################################################
            Zentriertes_df.plot( linestyle='-', use_index=True, figsize=(10,6))
            plt.title(f'Blech {BlechNummer} Übersicht Y zentrierte Kräfte ', fontsize=15)
            plt.xlabel('Datenpunkte', fontsize=14, labelpad=12)
            plt.ylabel('Kraft in [N]', fontsize=14, labelpad=12)
            plt.legend(loc='upper right', fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            #plt.show()
            plt.savefig(os.path.join(Sicherungsordner, f'{BlechNummer}_Plot_ÜbersichtZentrierungY.svg'), format='svg')
            plt.savefig(os.path.join(Sicherungsordner, f'{BlechNummer}_Plot_ÜbersichtZentrierungY.png'), format='png')
            plt.close()
            ########################################   PLOT Zentrierung X und Y mit gesamten Richtbereich jedoch ohne Einlauf #####################################################
            Preprocessed_Data.plot( linestyle='-', use_index=True, figsize=(10,6))
            plt.title(f'Blech {BlechNummer} Übersicht zentrierte Kräfte und ohne Offset',fontsize=15, pad=12)
            plt.xlabel('Datenpunkte', fontsize=14, labelpad=12)
            plt.ylabel('Kraft in [N]', fontsize=14, labelpad=12)
            plt.legend(loc='upper right',fontsize=12)
            plt.grid(True)
            plt.xticks(np.arange(0,8000,200), fontsize=4)
            plt.tight_layout()
            #plt.show()
            plt.savefig(os.path.join(Sicherungsordner, f'{BlechNummer}_Plot_Übersicht zentrierte Kräfte in X und Y.svg'), format='svg')
            plt.savefig(os.path.join(Sicherungsordner, f'{BlechNummer}_Plot_Übersicht zentrierte Kräfte in X und Y.png'), format='png')
            plt.close()
            
            ########################################   PLOT Zentrierung X und Y und stationärer Bereich #####################################################
            Data_stat.plot( linestyle='-', use_index=True, figsize=(10,6))
            plt.title(f'Stationärer Datenbereich für Blech {BlechNummer}', fontsize=15, pad=12)
            plt.xlabel('Datenpunkte', fontsize=14, labelpad=12)
            plt.ylabel('Kraft in [N]', fontsize=14, labelpad=12)
            plt.legend(loc='upper right',fontsize=12)
            plt.grid(True)
            plt.xticks(np.arange(0,2000,200), fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            #plt.show()
            plt.savefig(os.path.join(Sicherungsordner, f'{BlechNummer}_Plot_Preprocessed.svg'), format='svg')
            plt.savefig(os.path.join(Sicherungsordner, f'{BlechNummer}_Plot_Preprocessed.png'), format='png')
            plt.close()

        return Data_stat, Data_zentriert, BlechNummer
    # Falls eine Blechnummer aufgerufen wird, welche in den CSV Dateien fehlerhaft ist und somit durch die Read_Single_Blech Funktion nicht aufgerufen wird 
    else:
        return None, None, BlechNummer
    
