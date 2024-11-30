# Diese Funktion wird für das Vorbereiten der Daten auf die Deep Learning Modelle verwendet 

# Dafür werden die reingeladenen daten aus der Data_for_Modelling Datei in ein Notebook geladen und dieser Funktion in Form von data übergeben 
# In dieser Funktion werden dann die Zeitreihendaten gebildet, die Daten nach den beiden Evaluationsmetriken aufgeteilt und standardisiert

from PreProcessing_SingleBlech_Function import Preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random as rnd


# Interpolation = 0: Ohne Interpolierte Daten
# Interpolation = 1: Mit interpolierten Daten ---> Dann muss das interpolierte dataframe mit übergeben werden

# Train_Test_Split =1: Standardmäßige Aufteilung der Daten
# Train_Test_Split =2: Aufteilung nach einzelnen Blechen (neue Evaluationsmetrik)

# Window Size muss definiert werden, und kann variabel angepasst werden

# Zufallsfaktor (random) für Reproduzierbarkeit definieren

# size: gibt die Größe an Validierungs- und Testdaten an, welche wiederum 50/50 geteilt werden: 0.2 --> 80% Training 10% Validierungs 10% Testdaten

# Validation_data != 1: Keine Validationsdaten, dadurch size =0.2 20% Test und 80% Training
# Validation_data  = 1: Validationsdaten werden erstellt, siehe oben size 

# Scaler: Scaler kann beliebig varriert werden, Standard: StandardScaler

# Datengröße muss immer mit angegebn werden, wenn Bereich im Preprocessing variiert wird hier mit beachten

def Fensterung_Scale(data, interpoliertesdf=None, window_size = 10, Datengröße = 1800, size=0.2, random =1, Scaler = StandardScaler, Train_Test_Split =2, Validation_data=1, Interpolation=0):
    
    # Labels die gedroppt werden später
    Columns_drop = ['X_opt-X-Ist','Y_Opt-Y_ist','phi_Opt-phi_ist']

    Dataframe_Features = []
    Dataframe_Labels = []

    # Berechne Fensteranzahl aus Datengröße (1800) und festgelegter Window Size
    Fensteranzahl = Datengröße - window_size

    # Funktion um aus dem Dataframe die Features und Labels in Form von Zeitreihendaten zu erstellen
    def create_windows(data_features, data_labels, window_size):
        # Leere Listen erstellen zum appenden
        windows_X = []
        windows_Y = []
        
        # Iteration über die Länge der Features für Windows
        for i in range(len(data_features) - window_size):
            #print(i)
            windows_X.append(data_features[i:i + window_size])
            windows_Y.append(data_labels[i + window_size -1: i + window_size])
            
        return np.array(windows_X), np.array(windows_Y)
    
    #Anzahl an Bleche in der Gesamtmatrix
    BlechAnzahl_Test = int(len(data)/Datengröße)
    print(f'Anzahl an Bleche in der Gesamtmatrix {BlechAnzahl_Test}')

    # Über die Anzahl an Blechen in den Daten iterieren um die Fenster der einzelnen Bleche zu erstellen 
    for n in range(BlechAnzahl_Test):
        #print(n)
        # Extrahiere jedes einzelne Blech aus dem gesamten Dataframe aller Bleche
        einzelnes_Blech = data.iloc[n*Datengröße:(n+1)*Datengröße]
        ##print(einzelnes_Blech)
        # Aufteilung in Features und Labels
        X = einzelnes_Blech.drop(columns=Columns_drop)
        Y = einzelnes_Blech[Columns_drop]
        
        # Aufruf der Funktion für die Erstellung der Fenster
        Window_Features, Window_Labels = create_windows(X, Y, window_size)
      
        Dataframe_Features.append(Window_Features)
        Dataframe_Labels.append(Window_Labels)
        
    # Zusammenfügen der Listen zu einem Array         
    Merged_Features = np.vstack(Dataframe_Features)
    Merged_labels = np.vstack(Dataframe_Labels)
    #print(len(Merged_Features))
    
    # Falls standardmäßige Aufteilung der Daten ausgewählt wird
    if Train_Test_Split == 1:
        
        X_test_interpolation = None
        Y_test_interpolation = None
        angepasste_Blechnummern = None
        Angepasste_Blechnummern_test = None
        Blechnummern_Test = None
        #Um keinen Fehler bei der Rückgabe für fehlende Daten zu erhalten (kann auch durch weiter if Abfrage beim retrun implementiert werden)
        
        # Train-Test Split in Trainings und Testdaten
        X_train, X_test, Y_train, Y_test = train_test_split(Merged_Features, Merged_labels, test_size= size, random_state= random)
        
        # Falls zusätzlich Validierungsdaten benötigt, weitere Aufteilung der Testdaten in Validierung und Testdaten
        if Validation_data == 1: 
            X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size= 0.5, random_state= random)    
            
        else:
            X_val = Y_val = None
          
    # Falls einzelne Bleche extrahiert werden sollen für die Testdaten (neue Evaluationsmetrik) 
    elif Train_Test_Split == 2:
        
        # Leere Listen
        Liste_Features_val = []
        Liste_Labels_val = []
        Liste_Features_test = []
        Liste_Labels_test = []
        Liste_Features_Test_interpolation = []
        Liste_Labels_Test_interpolation = []
        
        # Falls Interpoliert wird
        if Interpolation == 1:
            
            # Falls falsche Einstellung der Funktion übergeben wird
            if interpoliertesdf is None:
                print('Interpoliertes Dataframe muss vorgegeben werden, da Interpolation auf 1 gestellt ist. Wenn keine Interpolierten Daten eingegeben werden, Interpolation auf 0 stellen oder aus dem Funktionsaufruf löschen')
            
            # Ursprüngliche Blechzahl für die im Versuch aufgenommenen Daten berechnen
            else:
                # Ursprüngliche Anzahl an Blechen im Dataframe
                Blechanzahl_old = len(Merged_Features)/Fensteranzahl-len(interpoliertesdf)/Datengröße
                #print(Blechanzahl_old)
                
                #Anzahl an Bleche in der Gesamtmatrix
                BlechAnzahl_ges = len(Merged_Features)/Fensteranzahl
                
                #Wie viele Bleche für die Testdaten extrahiert werden sollen für die reinen Versuchsdaten
                Test_Blechmenge_old = round(Blechanzahl_old * size) 
                
                #Wie viele Bleche für die Testdaten extrahiert werden sollen für die reinen interpolierten Daten
                Test_Blechmenge_ges = round((BlechAnzahl_ges-Blechanzahl_old) * size)
                #print(Test_Blechmenge_old)
                #print(Test_Blechmenge_ges)
                
                # Auswahl von Random Blechen basierend auf dem Zufallsfaktor random
                rnd.seed(random)
                Random_Blech_old = rnd.sample(range(1, int(Blechanzahl_old)),int(Test_Blechmenge_old))
                Random_Blech_ges = rnd.sample(range(int(Blechanzahl_old), int(BlechAnzahl_ges)),int(Test_Blechmenge_ges))
                # Random Bleche für alle Daten als Interpoliert + Versuchsdaten
                Random_Blech = Random_Blech_old + Random_Blech_ges
                print(f'Random Blech Nummern für Testdaten der Versuche {Random_Blech_old}')
                print(f'Random Blech Nummern für Testdaten aller Daten mit interpolierten Werten {Random_Blech_ges}')
                print(len(Random_Blech))
                
        else:
            
            BlechAnzahl = len(Merged_Features)/Fensteranzahl #Anzahl an Bleche in der Gesamtmatrix
            print(BlechAnzahl)
            Test_Blechmenge = round(BlechAnzahl * size) #Wie viele Bleche für die Testdaten extrahiert werden sollen
            #print(Test_Blechmenge)
            rnd.seed(random)
            Random_Blech = rnd.sample(range(1, int(BlechAnzahl)),int(Test_Blechmenge)) # Erstelle Random Blech Nummern, welche je nachdem welcher random state vorgegeben wird, für die Testdaten sind
            #print([x + 12 for x in Random_Blech]) #Für die Angabe, welche der Bleche für den Test verwendet werden --> Wird in der Modellierung Datei beim Aufrufen geprinted
        
        # Umrechnung der Blechnnummern für die genaue Zuordnung, Random_Blech ist lediglich um die richtigen Bleche im Dataframe zu adressieren, da manche Bleche fehlerhaft sind und rausgelassen wurden (insgesamt 142 Bleche)
        # Hiermit werden die Nummern zur Adressierung im Dataframe umgerechnet in die wahren Blechnummern für eine genaue Zuordnung 
        def Blechnummer_Umrechner(Random_Blech):
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
                
            return angepasste_Blechnummern
        
        # Aufteilung der rausgezogenen Bleche in Test und Validation Data für die Hyperparametervalidierung 
        if Validation_data == 1: 
            
            # Aufteilung der Bleche in Validation und Testbleche
            Random_Blech_val = Random_Blech[:len(Random_Blech)//2]
            Random_Blech_test = Random_Blech[len(Random_Blech)//2:]
            # Umrechnung der Bleche auf wahre Blechnummern
            angepasste_Blechnummern = Blechnummer_Umrechner(Random_Blech)
            Angepasste_Blechnummern_test = angepasste_Blechnummern[len(angepasste_Blechnummern)//2:]
            Angepasste_Blechnummern_val = angepasste_Blechnummern[:len(angepasste_Blechnummern)//2]
            print(f'Richtige Blechnummern umgerechnet der Validierungsdaten {Angepasste_Blechnummern_val}')
            print(f'Richtige Blechnummern umgerechnet der Testdaten {Angepasste_Blechnummern_test}')
            print(f'Blechnummern für Validierungsdaten, abgezählt vom Array nicht die Originaldaten {Random_Blech_val}')
            print(f'Blechnummern für Testdaten, abgezählt vom Array nicht die Originaldaten{Random_Blech_test}')
        
            # Iteriere über die Blechnummern, um die Daten für die Validation zu extrahieren (Gesamtdaten mit Interpolation)
            for i in Random_Blech_val:
                single_Features_val = Merged_Features[Fensteranzahl*i:Fensteranzahl*(i+1)]
                Single_Labels_val = Merged_labels[Fensteranzahl*i:Fensteranzahl*(i+1)]
                #print(single_Features)
                Liste_Features_val.append(single_Features_val)
                Liste_Labels_val.append(Single_Labels_val)
                
            # Iteriere über die Blechnummern, um die Daten für den Testsplit zu extrahieren (Gesamtdaten mit Interpolation)
            for i in Random_Blech_test:
                single_Features_test = Merged_Features[Fensteranzahl*i:Fensteranzahl*(i+1)]
                Single_Labels_test = Merged_labels[Fensteranzahl*i:Fensteranzahl*(i+1)]
                #print(single_Features)
                Liste_Features_test.append(single_Features_test)
                Liste_Labels_test.append(Single_Labels_test)
            
            # Falls Interpoliert wird, werden hier zusätzlich die Daten noch in ausschließlich Versuchsdaten aufgeteilt zusätzlich zu den Gesamtdaten oben
            if Interpolation ==1:
                
                # Erstellung der Blechnummern für die reinen Versuchsdaten
                Random_Blech_val_Original = Random_Blech_old[:len(Random_Blech_old)//2]
                Random_Blech_test_Original = Random_Blech_old[len(Random_Blech_old)//2:]
                # Neue Liste mit Werten unter 142 für den Fall das eine Interpolation durchgeführt wird (Umrechnen auf Originale Bleche aus dem Versuch)
                Blechnummern_Val_Int = Blechnummer_Umrechner(Random_Blech_val_Original)
                Blechnummern_Test_Int = Blechnummer_Umrechner(Random_Blech_test_Original)
                print(f'Blechnummern für Testdaten der aufgenommenen Daten ohne die interpolierten Werte {Blechnummern_Test_Int}')
                
                # Iteration über die Bleche für die reinen Versuchsdaten
                for z in Random_Blech_test_Original:
                    single_Features_test_Original = Merged_Features[Fensteranzahl*z:Fensteranzahl*(z+1)]
                    Single_Labels_test_Original = Merged_labels[Fensteranzahl*z:Fensteranzahl*(z+1)]
                    
                    Liste_Labels_Test_interpolation.append(Single_Labels_test_Original)
                    Liste_Features_Test_interpolation.append(single_Features_test_Original)
                    
                # Zusammenfügen der Liste der Testdaten für die Validierung der Interpolation zu einem gesamten Array     
                X_test_interpolation = np.vstack(Liste_Features_Test_interpolation)
                Y_test_interpolation =np.vstack(Liste_Labels_Test_interpolation)
                print(Y_test_interpolation.shape)
                
            # Zusammenfügen der Liste der Validierungsdaten zu einem gesamten Array     
            X_val = np.vstack(Liste_Features_val)
            Y_val = np.vstack(Liste_Labels_val)
            
            # Mische die Validationsdaten zufällig
            np.random.seed(random)
            indices_val = np.random.permutation(X_val.shape[0])
            print(X_val.shape[0])
            X_val = X_val[indices_val]
            Y_val = Y_val[indices_val]
            print(f'Shape nach dem Random Sampling des Arrays von X_val: {X_val.shape}')
            
            # Zusammenfügen der Liste der Testdaten zu einem gesamten Array     
            X_test = np.vstack(Liste_Features_test)
            Y_test = np.vstack(Liste_Labels_test)
            
            # Entfernen der Testdaten und Validierungsdaten aus den Trainingsdaten 
            X_train = np.delete(Merged_Features, [np.arange(int(Fensteranzahl * i), int(Fensteranzahl * (i + 1))) for i in Random_Blech], axis=0)
            Y_train = np.delete(Merged_labels, [np.arange(int(Fensteranzahl * i), int(Fensteranzahl * (i + 1))) for i in Random_Blech], axis=0)
        
        # Falls keine Validierungsdaten benötigt werden
        # Iteriere über die Blechnummern, um die Daten für den Testsplit zu extrahieren
        else:
            
            X_val = Y_val = None
            
            for i in Random_Blech:
                # Aufteilung in Features und Labels
                single_Features = Merged_Features[Fensteranzahl*i:Fensteranzahl*(i+1)]
                Single_Labels = Merged_labels[Fensteranzahl*i:Fensteranzahl*(i+1)]
                # Umrechnen der Blechnummern zu den Originalen Blechnummern
                Blechnummern_Test = Blechnummer_Umrechner(Random_Blech)
                #print(single_Features)
                Liste_Features_test.append(single_Features)
                Liste_Labels_test.append(Single_Labels)
            
            # Für Interpolation wieder zusätzlich nur die Testdaten extrahieren
            if Interpolation ==1:
                # Neue Liste mit Werten unter 142 für den Fall das eine Interpolation durchgeführt wird
                #Blechnummern_Test_Array = [x for x in Random_Blech if x < 143]
                angepasste_Blechnummern = Blechnummer_Umrechner(Random_Blech)
                Blechnummern_Test_original = Blechnummer_Umrechner(Random_Blech_old)
                
                for z in Random_Blech_old:
                    single_Features_test_Original = Merged_Features[Fensteranzahl*z:Fensteranzahl*(z+1)]
                    Single_Labels_test_Original = Merged_labels[Fensteranzahl*z:Fensteranzahl*(z+1)]
                    
                    Liste_Labels_Test_interpolation.append(Single_Labels_test_Original)
                    Liste_Features_Test_interpolation.append(single_Features_test_Original)
                    
                print(f'Richtige Blechnummern umgerechnet der gesamten Testdaten {angepasste_Blechnummern}')
                print(f'Richtige Blechnummern umgerechnet der Testdaten für die Interpolierten Daten, welche wirklich gemessen wurden {Blechnummern_Test_original}')
                
                # Zusammenfügen der Liste der Testdaten für die Validierung der Interpolation zu einem gesamten Array     
                X_test_interpolation = np.vstack(Liste_Features_Test_interpolation)
                Y_test_interpolation =np.vstack(Liste_Features_Test_interpolation)
            
            # Zusammenfügen der Liste zu einem gesamten Array     
            X_test = np.vstack(Liste_Features_test)
            Y_test = np.vstack(Liste_Labels_test)     

            # Entfernen der Testdaten aus den Trainingsdaten
            X_train = np.delete(Merged_Features, [np.arange(int(Fensteranzahl * i), int(Fensteranzahl * (i + 1))) for i in Random_Blech], axis=0)
            Y_train = np.delete(Merged_labels, [np.arange(int(Fensteranzahl * i), int(Fensteranzahl * (i + 1))) for i in Random_Blech], axis=0)
            
        # Mische die Trainingsdaten und Testdaten zufällig
        np.random.seed(random)
        indices_train = np.random.permutation(X_train.shape[0])
        indices_test = np.random.permutation(X_test.shape[0])
        
        #print(X_train.shape[0])
        X_train = X_train[indices_train]
        Y_train = Y_train[indices_train]
       
        X_test = X_test[indices_test]
        Y_test = Y_test[indices_test]
       
        # print(f'Shape nach dem Random Sampling des Arrays von X_train: {X_train.shape}')
        # print(f'Shape nach dem Random Sampling des Arrays von Y_train: {Y_train.shape}')
        
        # Falls Interpoliert wird mische die Interpolierten Daten 
        if Interpolation ==1:
            indices_int = np.random.permutation(X_test_interpolation.shape[0])
            X_test_interpolation = X_test_interpolation[indices_int]
            Y_test_interpolation = Y_test_interpolation[indices_int]
            #print(f'Shape nach dem Random Sampling des Arrays von Testdaten in den Interpolierten Daten: {Y_test_interpolation.shape}')
        
    else:
        raise ValueError("Fehlerhafte Variable für Train_Test_Split. Muss 1 oder 2 gewählt werden.")
        
    # Normalisieren der Daten
    # Definiere die Indizes für jede Gruppe von Merkmalen, Unterscheidung in Kräfte und der beiden unterschiedlichen Positionen
    group_features = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9], [10]]
    group_labels = [[0,1], [2]]

    # Definiere den Skalierer
    scalers_features = [Scaler() for _ in group_features]
    scaler_labels = [Scaler() for _ in group_labels]

    # Liste für skalierte Werte 
    X_train_scaled_list = []
    X_val_scaled_list = []
    X_test_scaled_list = []
    Y_train_scaled_list = []
    Y_val_scaled_list = []
    Y_test_scaled_list = []
    X_test_scaled_list_int =[]
    Y_test_scaled_list_int =[]
    
    # Skaliere jede Gruppe der Features separat in einer Schleife
    for indices, scaler in zip(group_features, scalers_features):
        # Daten müssen umgeformt werden von einem Array in die gewünschte Dimension
        X_train_group = X_train[:, :, indices].reshape(-1, len(indices))
        X_test_group = X_test[:, :, indices].reshape(-1, len(indices))
        
        # Fit und Transform auf die Trainingsdaten und nur Transform auf die Testdaten
        X_train_scaled_group = scaler.fit_transform(X_train_group)
        X_test_scaled_group = scaler.transform(X_test_group)
            
        # Forme die skalierten Daten zurück in das ursprüngliche Shape
        X_train_scaled_group = X_train_scaled_group.reshape(len(X_train), -1, len(indices))
        X_test_scaled_group = X_test_scaled_group.reshape(len(X_test), -1, len(indices))

        X_train_scaled_list.append(X_train_scaled_group)
        X_test_scaled_list.append(X_test_scaled_group)
        
        # Falls Interpoliert wird müssen diese auch transformiert werden
        if Interpolation == 1 and Train_Test_Split ==2:
        
            X_test_group_int = X_test_interpolation[:, :, indices].reshape(-1, len(indices))
            X_test_scaled_group_int = scaler.transform(X_test_group_int)
            X_test_scaled_group_int = X_test_scaled_group_int.reshape(len(X_test_interpolation), -1, len(indices))
            X_test_scaled_list_int.append(X_test_scaled_group_int)
        
        # Zusätzlich die validierten Daten transformieren
        if Validation_data ==1:
            X_val_group = X_val[:, :, indices].reshape(-1, len(indices))
            X_val_scaled_group = scaler.transform(X_val_group)
            X_val_scaled_group = X_val_scaled_group.reshape(len(X_val), -1, len(indices))
            X_val_scaled_list.append(X_val_scaled_group)
    
    # Skaliere jede Gruppe der Labels separat in einer Schleife
    for indices, scaler in zip(group_labels, scaler_labels):
        # print(indices)
        # print(scaler)
        # In das richtige Format bringen, um die Skalierung durchführen zu können 
        Y_train_group = Y_train[:, :, indices].reshape(-1, len(indices))
        Y_test_group = Y_test[:, :, indices].reshape(-1, len(indices))
        #print(Y_test_group.shape)
        # Durchführung der Skalierung
        Y_train_scaled_group = scaler.fit_transform(Y_train_group)
        Y_test_scaled_group = scaler.transform(Y_test_group)
        #print(Y_test_scaled_group.shape)
        
        # Forme die skalierten Daten zurück in das ursprüngliche Shape
        Y_train_scaled_group = Y_train_scaled_group.reshape(len(Y_train), -1, len(indices))
        Y_test_scaled_group = Y_test_scaled_group.reshape(len(Y_test), -1, len(indices))
        #print(Y_test_scaled_group.shape)
        Y_train_scaled_list.append(Y_train_scaled_group)
        Y_test_scaled_list.append(Y_test_scaled_group)
        
        # Falls Interpoliert wird müssen diese auch transformiert werden
        if Interpolation ==1 and Train_Test_Split==2:
        
            Y_test_group_int = Y_test_interpolation[:, :, indices].reshape(-1, len(indices))
            Y_test_scaled_group_int = scaler.transform(Y_test_group_int)
            Y_test_scaled_group_int = Y_test_scaled_group_int.reshape(len(Y_test_interpolation), -1, len(indices))
            Y_test_scaled_list_int.append(Y_test_scaled_group_int)
            #print(Y_test_scaled_list)
        
        # Zusätzlich die validierten Daten transformieren
        if Validation_data ==1:
            Y_val_group = Y_val[:, :, indices].reshape(-1, len(indices))
            Y_val_scaled_group = scaler.transform(Y_val_group)
            Y_val_scaled_group = Y_val_scaled_group.reshape(len(Y_val), -1, len(indices))
            Y_val_scaled_list.append(Y_val_scaled_group)

    # Überprüfe die Dimensionen der skalierten Feature-Arrays
    # print([arr.shape for arr in X_train_scaled_list]) 
    # print([arr.shape for arr in X_test_scaled_list])
    # print([arr.shape for arr in Y_train_scaled_list])
    #print([arr.shape for arr in Y_test_scaled_list_int])
    # if Validation_data ==1:
    #     print([arr.shape for arr in X_val_scaled_list])

    # Füge die skalierten Features wieder zusammen
    X_train_scaled = np.concatenate(X_train_scaled_list, axis=2)
    X_test_scaled = np.concatenate(X_test_scaled_list, axis=2)
    
    Y_train_scaled = np.concatenate(Y_train_scaled_list, axis=2)
    Y_test_scaled = np.concatenate(Y_test_scaled_list, axis=2)
    print(Y_test_scaled.shape)
    
    # Interpolierten Daten der saklierten Fenster wieder zusammenfügen
    if Interpolation ==1 and Train_Test_Split==2:
    
        X_test_scaled_int = np.concatenate(X_test_scaled_list_int, axis=2)
        Y_test_scaled_int = np.concatenate(Y_test_scaled_list_int, axis=2)
        
        print(f'Shape für die Labels der Testdaten nur mit Blechen aus dem Versuch {Y_test_interpolation.shape}')
        print(f'Shape für die Features der Testdaten nur mit Blechen aus dem Versuch {X_test_interpolation.shape}')
        
    # Überprüfe die Dimensionen der Labels und Features für Training und Testdaten
    print(f'Shape für die Features der gesamten Trainingsdaten, also im Falle einer Interpolation mit allen Daten {X_train_scaled.shape}') 
    print(f'Shape für die Features der gesamten Testdaten, also im Falle einer Interpolation mit allen Daten {X_test_scaled.shape} ')
    print(f'Shape für die Labels der gesamten Trainingsdaten, also im Falle einer Interpolation mit allen Daten {Y_train_scaled.shape}')
    print(f'Shape für die Labels der gesamten Testdaten, also im Falle einer Interpolation mit allen Daten {Y_test_scaled.shape}')
  
    # Rückgaben für die unterschiedlichen Fallunterscheiden
    # Beim Aufrufen der FUnktion muss auf die richtige Anzahl an übergebenden Variablen geachtet werden
    if Validation_data == 1 and Interpolation != 1:
        X_val_scaled = np.concatenate(X_val_scaled_list, axis=2)
        Y_val_scaled = np.concatenate(Y_val_scaled_list, axis=2)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, Angepasste_Blechnummern_test
    
    elif Validation_data ==1 and Interpolation ==1 and Train_Test_Split==2:
        
        X_val_scaled = np.concatenate(X_val_scaled_list, axis=2)
        Y_val_scaled = np.concatenate(Y_val_scaled_list, axis=2)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, X_test_scaled_int, Y_test_interpolation, Blechnummern_Test_Int
    
    elif Validation_data ==1 and Interpolation ==1 and Train_Test_Split==1:
        
        X_val_scaled = np.concatenate(X_val_scaled_list, axis=2)
        Y_val_scaled = np.concatenate(Y_val_scaled_list, axis=2)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels
    
    elif Validation_data !=1 and Interpolation ==1:
    
        return X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_train, Y_test, scalers_features, scaler_labels, X_test_scaled_int, Y_test_interpolation, Blechnummern_Test_original
    
    else:
        
        return X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_train, Y_test, scalers_features, scaler_labels, Blechnummern_Test