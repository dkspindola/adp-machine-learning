#Imports für Modellbildung
import pandas as pd
from Load_Data_for_Modelling import Get_data
from Splitting_Scaling_Function import Split_Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

#Imports für ML-Validierung
import pandas as pd
import random
#from Load_Data_for_Modelling_Function import Data_for_Model
from Load_Data_for_Modelling import Get_data

#Imports für error_conclusion und error_analysis
import pandas as pd
import numpy as np

#Imports für Funktion feature_importance
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd 




def ML(data, ML_Model= 1, x_start=0, x_end= 1800, size=0.2, Train_Test_Split=2, Datengröße=1800, random=42):
    """
    Parameter:
    data: Die Daten, die für das Modell verwendet werden.
    ML_Model: Der zu verwendende Modelltyp (1=LinearRegression, 2=RandomForest, etc.).
    x_start: Startindex für die Daten.
    x_end: Endindex für die Daten.
    size: Anteil der Testdaten.
    Train_Test_Split: Methode zum Teilen der Daten.
    Datengröße: Größe der Daten.
    random: Zufallsseed.
    """

    #Daten laden über main mit Variable data
    #Für den Durchlauf mit der Split Scalling Function der Features
    X_train, X_val, X_test, Y_train, Y_val, Y_test = Split_Scaling(data, size=size, Train_Test_Split=Train_Test_Split, random=random, Validation_Data=1, Datengröße=Datengröße)
    
    if ML_Model == 1:
        Model = LinearRegression()
        model_name = 'Lineare Regression'
        
    elif ML_Model ==2:
        pre_process = PolynomialFeatures(degree=3)

        X_poly_train = pre_process.fit_transform(X_train)
        X_poly_test = pre_process.fit_transform(X_test)
        
        Model = LinearRegression()
        
        model_name = 'Polynominale Regression'
        
    elif ML_Model ==3:
        Model = DecisionTreeRegressor(criterion= 'absolute_error', max_depth=5)
        
        model_name = 'Decision Tree Regression' 
    
    elif ML_Model ==4:
        Model = RandomForestRegressor(n_estimators=1400, max_depth=70, min_samples_split=10, min_samples_leaf=1, max_features='log2', bootstrap=False)
        
        model_name = 'Random Forest Regression'
        
    elif ML_Model ==5:
        Model = KNeighborsRegressor(algorithm= 'ball_tree', n_neighbors= 3, weights= 'distance', metric='manhattan')
        model_name = 'KNeighbors Regression'

    elif ML_Model ==6:
        Model_einzel = SVR(kernel='rbf', gamma=0.2682695795279725, epsilon=0.1, C=51.7947467923121)

        Model = MultiOutputRegressor(Model_einzel)
        
        model_name = 'Support Vector Regression'
        
    if ML_Model ==2:
        
        Model.fit(X_poly_train, Y_train)
        
        Y_pred_train = Model.predict(X_poly_train)
        Y_pred_test = Model.predict(X_poly_test)
        
    else:
        Model.fit(X_train, Y_train)
        
        Y_pred_train = Model.predict(X_train)
        Y_pred_test = Model.predict(X_test)

    Predictions = pd.DataFrame({'Predicted_X_opt-X-Ist': Y_pred_test[:, 0],'Predicted_Y_Opt-Y_ist': Y_pred_test[:, 1],'Predicted_phi_Opt-phi_ist': Y_pred_test[:,2]})
    
    X_p = Y_pred_test[:, 0]
    Y_p = Y_pred_test[:, 1]
    Phi_p = Y_pred_test[:, 2]
    
    print(X_p.shape)
    print(Y_pred_test.shape)
    print(Y_test.shape)
     # Error für jedes Label berechnen
    Fehler_X = Y_test.iloc[:,0]-X_p
    Fehler_Y = Y_test.iloc[:,1]-Y_p
    Fehler_Phi = Y_test.iloc[:,2]-Phi_p
    
    
    # Fehler in einen DataFrame konvertieren
    df_Fehler = pd.DataFrame({ 
        'Label_X': Y_test.iloc[:,0],
        'Label_Y': Y_test.iloc[:,1],
        'Label_Phi': Y_test.iloc[:,2],
        'Fehler_X': Fehler_X,
        'Fehler_Y': Fehler_Y,
        'Fehler_Phi': Fehler_Phi})
    #Hier noch die Blechnnummer hinzufügen, in dem ich die Blechnnummern über die for Schleife in der Split Scaling Funktion in der selben Datengröße erstelle wie die Y_test und Y_val Daten 
    # Dann kann man diese hier einfach hinzufügen 

    MAE_train = mean_absolute_error(Y_train, Y_pred_train, multioutput='raw_values')
    MAE_test = mean_absolute_error(Y_test, Y_pred_test, multioutput='raw_values')
    M2E_train = mean_squared_error (Y_train, Y_pred_train, multioutput='raw_values')
    M2E_test = mean_squared_error (Y_test, Y_pred_test, multioutput='raw_values')

    print(f'MAE für jeden Output der Trainingsdaten für {model_name}: {MAE_train}') 
    print(f'MAE für jeden Output der Testdaten für {model_name}: {MAE_test}')
    print(f'M2E für jeden Output der Trainingsdaten für {model_name}: {M2E_train}') 
    print(f'M2E für jeden Output der Testdaten für {model_name}: {M2E_test}')
    
    return Y_pred_train, Y_pred_test, MAE_train, MAE_test, M2E_train, M2E_test, model_name, Model, df_Fehler

#Validierung der Modelle
def Validierung_ML(data, modelle, random_seed_list, x_start=0, x_end=1800):
    '''Parameter:
    data: Die Daten, die für das Modell verwendet werden.
    modelle: Eine Liste der Modellnummern, die evaluiert werden sollen.
    random_seed_list: Eine Liste von zufälligen Seed-Werten für die Cross-Validation.
    x_start: Der Startindex für die Daten.
    x_end: Der Endindex für die Daten.
    '''
    #Daten laden über main mit Variable data
    
    # Erstelle Dataframes für die gesamten Fehler aller Modelle die dann erweitert werden
    MAE_StandardSplit_df = pd.DataFrame(columns=['Model','CV', 'Datentyp','Error', 'X', 'y', 'phi'])
    M2E_StandardSplit_df = pd.DataFrame(columns=['Model','CV', 'Datentyp','Error', 'X', 'y', 'phi'])
    MAE_BlechSplit_df = pd.DataFrame(columns=['Model','CV', 'Datentyp','Error', 'X', 'y', 'phi'])
    M2E_BlechSplit_df = pd.DataFrame(columns=['Model','CV', 'Datentyp','Error', 'X', 'y', 'phi'])

    # Initialisierung der Variablen für Fehlerdaten
    Df_Fehler_append_Standardsplit = pd.DataFrame()
    Df_Fehler_append_Blechsplit = pd.DataFrame()

    # Schleife für Standard Split
    for i in modelle:
        
        for n in Random_numbers:
            
            # Aufruf der Funktion mit dem entsprechenden Modell und dem entsprechenden Zufallsfaktor für Standard Split
            Y_pred_train, Y_pred_test, MAE_train, MAE_test, M2E_train, M2E_test, model_name, Model, df_Fehler = ML(data, ML_Model=i, size=0.2, Train_Test_Split=1, Datengröße=1800, random=n)
            
            # Hinzufügen der MAE-Werte zu den DataFrames
            MAE_StandardSplit_df = pd.concat([MAE_StandardSplit_df, pd.DataFrame([{'Model': model_name,'CV':n, 'Datentyp': 'Train', 'Error' : 'MAE', 'X': MAE_train[0], 'y': MAE_train[1], 'phi': MAE_train[2]}])], ignore_index=True)
            MAE_StandardSplit_df = pd.concat([MAE_StandardSplit_df, pd.DataFrame([{'Model': model_name,'CV':n, 'Datentyp': 'Test', 'Error' : 'MAE', 'X': MAE_test[0], 'y': MAE_test[1], 'phi': MAE_test[2]}])], ignore_index=True)
        
            # Hinzufügen der M2E-Werte zu den DataFrames
            M2E_StandardSplit_df = pd.concat([M2E_StandardSplit_df, pd.DataFrame([{'Model': model_name,'CV':n, 'Datentyp': 'Train', 'Error' : 'M2E', 'X': M2E_train[0], 'y': M2E_train[1], 'phi': M2E_train[2]}])], ignore_index=True)
            M2E_StandardSplit_df = pd.concat([M2E_StandardSplit_df, pd.DataFrame([{'Model': model_name,'CV':n, 'Datentyp': 'Test', 'Error' : 'M2E', 'X': M2E_test[0], 'y': M2E_test[1], 'phi': M2E_test[2]}])], ignore_index=True)
            
            # Füge das Modell und die Split Methode sowie den Zufallsfaktor der Liste der Fehler hinzu und appende diese in jedem Durchlauf
            df_Fehler.insert(loc=0, column='Modell', value=i)
            df_Fehler.insert(loc=1, column='SplitMethode', value='Standardsplit')
            df_Fehler.insert(loc=2, column='CV', value=n)
            Df_Fehler_append_Standardsplit = pd.concat([Df_Fehler_append_Standardsplit,df_Fehler], ignore_index=True)
            
    # Schleife für Blech Split
    for i in modelle:
        
        #Über jeden Zufallsfaktor iterieren
        for n in Random_numbers:
            
            # Aufruf der Funktion mit dem entsprechenden Modell und dem entsprechenden Zufallsfaktor für Blech Split
            Y_pred_train, Y_pred_test, MAE_train, MAE_test, M2E_train, M2E_test, model_name, Model, df_Fehler = ML(data, ML_Model=i, size=0.2, Train_Test_Split=2, Datengröße=1800, random=n)
            
            # Hinzufügen der MAE-Werte zu den DataFrames
            MAE_BlechSplit_df = pd.concat([MAE_BlechSplit_df, pd.DataFrame([{'Model': model_name,'CV':n, 'Datentyp': 'Train', 'Error' : 'MAE', 'X': MAE_train[0], 'y': MAE_train[1], 'phi': MAE_train[2]}])], ignore_index=True)
            MAE_BlechSplit_df = pd.concat([MAE_BlechSplit_df, pd.DataFrame([{'Model': model_name,'CV':n, 'Datentyp': 'Test', 'Error' : 'MAE', 'X': MAE_test[0], 'y': MAE_test[1], 'phi': MAE_test[2]}])], ignore_index=True)
            # Hinzufügen der M2E-Werte zu den DataFrames
            M2E_BlechSplit_df = pd.concat([M2E_BlechSplit_df, pd.DataFrame([{'Model': model_name,'CV':n, 'Datentyp': 'Train', 'Error' : 'M2E', 'X': M2E_train[0], 'y': M2E_train[1], 'phi': M2E_train[2]}])], ignore_index=True)
            M2E_BlechSplit_df = pd.concat([M2E_BlechSplit_df, pd.DataFrame([{'Model': model_name,'CV':n, 'Datentyp': 'Test', 'Error' : 'M2E', 'X': M2E_test[0], 'y': M2E_test[1], 'phi': M2E_test[2]}])], ignore_index=True)

            # Füge das Modell und die Split Methode sowie den Zufallsfaktor der Liste der Fehler hinzu und appende diese in jedem Durchlauf
            df_Fehler.insert(loc=0, column='Modell', value=i)
            df_Fehler.insert(loc=1, column='SplitMethode', value='Blechsplit')
            df_Fehler.insert(loc=2, column='CV', value=n)
            Df_Fehler_append_Blechsplit = pd.concat([Df_Fehler_append_Blechsplit,df_Fehler], ignore_index=True)
        
    # Ausgabe der DataFrames
    return (MAE_StandardSplit_df, M2E_StandardSplit_df, MAE_BlechSplit_df, M2E_BlechSplit_df, 
    Df_Fehler_append_Standardsplit, Df_Fehler_append_Blechsplit)

    print("MAE Standard Split:")
    print(MAE_StandardSplit_df)
    print("\nM2E Standard Split:")
    print(M2E_StandardSplit_df)
    print("\nMAE Blech Split:")
    print(MAE_BlechSplit_df)
    print("\nM2E Blech Split:")
    print(M2E_BlechSplit_df)


#  Diese Funktion speichert Fehlerdaten für MAE und M2E in CSV-Dateien für unterschiedliche Split-Methoden (Standard und BVlechsplit)
def error_conclusion(MAE_BlechSplit_df, MAE_StandardSplit_df, M2E_BlechSplit_df, M2E_StandardSplit_df, Df_Fehler_append_Standardsplit, Df_Fehler_append_Blechsplit, folder_path)
    '''Parameter:
    MAE_BlechSplit_df, MAE_StandardSplit_df, M2E_BlechSplit_df, M2E_StandardSplit_df : DataFrames -Die Fehlerdaten (MAE und M2E) für BlechSplit und StandardSplit.
    Df_Fehler_append_Standardsplit, Df_Fehler_append_Blech - split : DataFrames - Die vollständigen Fehlerdaten für die jeweiligen Split-Methoden.
    folder_path : Der Pfad, an dem die CSV-Dateien gespeichert werden sollen.
    '''

    # Umbennenung der Varibalen bzw. neue ABspeicherung dieser
    MAE_BlechSplit_df_Test = MAE_BlechSplit_df
    MAE_StandardSplit_df_Test = MAE_StandardSplit_df
    M2E_BlechSplit_df_Test = M2E_BlechSplit_df
    M2E_StandardSplit_df_Test = M2E_StandardSplit_df
    
    pd.set_option('display.precision', 10)
    
    # Zusammenfügen der Fehler von MAE und M2E für Standard Split und Blech Split
    Blechsplit_Errors_for_CSV = pd.concat([MAE_BlechSplit_df_Test, M2E_BlechSplit_df_Test], axis=1)
    Standardsplit_Errors_for_CSV = pd.concat([MAE_StandardSplit_df_Test, M2E_StandardSplit_df_Test], axis=1)
    Errors_for_CSV = pd.concat([Standardsplit_Errors_for_CSV, Blechsplit_Errors_for_CSV], axis= 0, ignore_index=True)

            
    print(Errors_for_CSV.columns)
    print(Errors_for_CSV['Model'])

    # Columns die in Strings für die CSV umgewandelt werden
    Errors_for_CSV.columns = ['Model', 'CV', 'Datentyp', 'Error', 'X', 'y', 'phi', 'Model1', 'CV1',
        'Datentyp1', 'Error1', 'X1', 'y1', 'phi1']

    # Umwandlung der Columns/Spalten in Strings für die bessere Darstellung in Excel
    for Column in Errors_for_CSV.columns:
            Errors_for_CSV[Column] = Errors_for_CSV[Column].astype(str).str.replace('.', ',')
            
    for Column in Df_Fehler_append_Standardsplit:
            Df_Fehler_append_Standardsplit[Column] = Df_Fehler_append_Standardsplit[Column].astype(str).str.replace('.', ',')

    for Column in Df_Fehler_append_Blechsplit:
            Df_Fehler_append_Blechsplit[Column] = Df_Fehler_append_Blechsplit[Column].astype(str).str.replace('.', ',')

    # Speichern der Daten in CSV-Dateien
    Errors_for_CSV.to_csv(f'{folder_path}\\Ergebnisse_BesteModelle_Konventionell_verschiedeneCVS_Random02.csv', index=True, sep=';')
    Df_Fehler_append_Standardsplit.to_csv(f'{folder_path}\\Fehler_Konventionelle_Modelle_best_Standardsplit__gesamteFehler_Random02.csv', index=True, sep=';')
    Df_Fehler_append_Blechsplit.to_csv(f'{folder_path}\\Fehler_Konventionelle_Modelle_best_Blechsplit__gesamteFehler_Random02.csv', index=True, sep=';')


def error_analysis(ordner: str, threshold: int = 100) -> pd.DataFrame:
    """
    Lädt eine CSV mit MAE-Daten, berechnet Mittelwerte und Standardabweichungen für Standard- und Blechsplit,
    ersetzt extreme Fehlerwerte über einem Schwellenwert, und speichert die Ergebnisse in einer neuen CSV.
    
    Parameters:
    - ordner (str): Ordnerpfad, in dem die CSV-Dateien gespeichert sind.
    - threshold (int): Schwellenwert, über dem Fehlerwerte ersetzt werden (Standard: 100).
    
    Returns:
    - pd.DataFrame: DataFrame mit den berechneten Mittelwerten und Standardabweichungen.
    """
    #Falls Änderungen an der Excel vorgenommen werden wollen wir die Excel mit den Ergebnissen der einzelnen CVS für Standard und Blechsplit hier reingeladen und die hohen Fehler der PR mit 10 ersetzt
    # Reinladen der gesamten Zufallsfaktoren und der Modelle um die Mittelwerte und Std zu berechnen
    
    # Reinladen der CSV mit den MAEs der verschiedenen Zufallsfaktoren
    MAE_ges =pd.read_csv(f'{ordner}\Ergebnisse_BesteModelle_Konventionell_verschiedeneCVS_Random02.csv', delimiter= ';')

    # Columns die in Floats umgewandelt werden
    columns_int = ['X', 'y', 'phi', 'X1', 'y1', 'phi1']
    # Umwandlung in Floats für Python
    for Column in columns_int:
            MAE_ges[Column] = MAE_ges[Column].astype(str).str.replace(',', '.')
            MAE_ges[Column] = MAE_ges[Column].astype(float)

    # Nehme nur die MAEs
    MAE_ges = MAE_ges.iloc[:,:8]

    # Grenze die Daten auf entsprechende MAEs ein
    MAE_ges = MAE_ges[MAE_ges['Datentyp'] == 'Test']
    MAE_StandardSplit_Test = MAE_ges.iloc[:40,:].copy()
    MAE_Blechsplit_Test = MAE_ges.iloc[40:,:].copy()

    # Datentypen von Test in entsprechend Blechsplit oder Standardsplit
    MAE_StandardSplit_Test['Datentyp'] = 'Standard Split'
    MAE_Blechsplit_Test['Datentyp'] = 'Blech Split'

    # Modelle für die Berechnung der Mittelwerte
    Liste_Modelle = ['Lineare Regression', 'Polynominale Regression', 'Random Forest Regression', 'KNeighbors Regression']

    # Neue Dataframes für Mean und Std
    MAE_Standardsplit_mean = pd.DataFrame(columns=['Model', 'Datentyp', 'Error', 'X', 'Y', 'Phi'])
    MAE_Blechsplit_mean = pd.DataFrame(columns=['Model', 'Datentyp', 'Error', 'X', 'Y', 'Phi'])
    MAE_Standardsplit_std = pd.DataFrame(columns=['Model', 'Datentyp', 'Error', 'X', 'Y', 'Phi'])
    MAE_Blechsplit_std = pd.DataFrame(columns=['Model', 'Datentyp', 'Error', 'X', 'Y', 'Phi']) 

    # Mittelwerte berechnen und in Form für Plots bringen
    # Treshhold wird über Input der Funktion definiert
    # Iteriere über die Modelle
    for i in Liste_Modelle:
        # Auswahl des entsprechenden Modells
        Standardsplit = MAE_StandardSplit_Test[MAE_StandardSplit_Test['Model'] == i].copy()
        Blechsplit = MAE_Blechsplit_Test[MAE_Blechsplit_Test['Model'] == i].copy()
        
        # Werte in den Spalten 'X', 'y' oder 'phi' ersetzen, wenn sie den Grenzwert überschreiten
        Standardsplit.loc[:,'X'] = np.where(Standardsplit['X'] > threshold, 10, Standardsplit['X'])
        Standardsplit.loc[:,'y'] = np.where(Standardsplit['y'] > threshold, 10, Standardsplit['y'])
        Standardsplit.loc[:,'phi'] = np.where(Standardsplit['phi'] > threshold, 10, Standardsplit['phi'])
        
        # Werte in den Spalten 'X', 'y' oder 'phi' ersetzen, wenn sie den Grenzwert überschreiten
        Blechsplit.loc[:,'X'] = np.where(Blechsplit['X'] > threshold, 10, Blechsplit['X'])
        Blechsplit.loc[:,'y'] = np.where(Blechsplit['y'] > threshold, 10, Blechsplit['y'])
        Blechsplit.loc[:,'phi'] = np.where(Blechsplit['phi'] > threshold, 10, Blechsplit['phi'])
        
        # Berechnung von Mittelwert Std für Blech und Standard Split
        Mean_Standard = Standardsplit[['X', 'y', 'phi']].mean()
        Mean_Blech = Blechsplit[['X', 'y', 'phi']].mean()
        Std_Standard = Standardsplit[['X', 'y', 'phi']].std()
        Std_Blech = Blechsplit[['X', 'y', 'phi']].std()
        
        # Zusammenfügen der Listen / MEan /Std
        MAE_Standardsplit_mean = pd.concat([MAE_Standardsplit_mean, pd.DataFrame([{'Model': i, 'Datentyp': 'Standard Split', 'Error': 'Mittelwert MAE', 'X': Mean_Standard['X'], 'Y': Mean_Standard['y'], 'Phi': Mean_Standard['phi']}])], ignore_index=True)
        MAE_Blechsplit_mean = pd.concat([MAE_Blechsplit_mean, pd.DataFrame([{'Model': i, 'Datentyp': 'Blech Split', 'Error': 'Mittelwert MAE', 'X': Mean_Blech['X'], 'Y': Mean_Blech['y'], 'Phi': Mean_Blech['phi']}])], ignore_index=True)
        MAE_Standardsplit_std = pd.concat([MAE_Standardsplit_std, pd.DataFrame([{'Model': i, 'Datentyp': 'Standard Split', 'Error': 'Standardabweichung MAE', 'X': Std_Standard['X'], 'Y': Std_Standard['y'], 'Phi': Std_Standard['phi']}])], ignore_index=True)
        MAE_Blechsplit_std = pd.concat([MAE_Blechsplit_std, pd.DataFrame([{'Model': i, 'Datentyp': 'Blech Split', 'Error': 'Standardabweichung MAE', 'X': Std_Blech['X'], 'Y': Std_Blech['y'], 'Phi': Std_Blech['phi']}])], ignore_index=True)
    
    # Über iterierte Dataframes zusammenfügen
    MAE_combi_Mittelwert = pd.concat([MAE_Standardsplit_mean, MAE_Blechsplit_mean], ignore_index=True)
    MAE_combi_Std = pd.concat([MAE_Standardsplit_std, MAE_Blechsplit_std], ignore_index=True)

    # Mean und Std Zusammenfügen
    MAE_Mittelwert_Std = pd.concat([MAE_combi_Mittelwert, MAE_combi_Std], axis=1, ignore_index=False)

    # Umwandlung der Punkte in Kommas für die Darstellung in der CSV Datei
    MAE_Mittelwert_Std = MAE_Mittelwert_Std.applymap(lambda x: str(x).replace('.', ','))

    print(MAE_Mittelwert_Std)
    # Speichere die Daten in einer CSV
    MAE_Mittelwert_Std.to_csv(f'{ordner}/Ergebnisse_besteModelle_Konventionell_Mittelwert_Std_Random002.csv', index=True, sep=';')


# Extrahieren der Feature-Importance und Visualisierung
def feature_importance(model, X_train, model_name: str, output_folder: str, color_thesis=(0, 90/255, 169/255), threshold: float = 0.0009):
    """  
    Parameter:
        model: Das trainierte Modell (z. B. ein Random Forest Regressor).
        X_train (pd.DataFrame): Trainingsdaten mit den Features.
        model_name (str): Name des Modells (für Dateinamen und Titel).
        output_folder (str): Pfad zum Ordner, in dem der Plot gespeichert wird.
        color_thesis (tuple): Farbe für die Balken im Diagramm (RGB-Tuple).
        threshold (float): Schwelle, unter der Feature-Wichtigkeiten als 0 dargestellt werden.
        """

    feature_importance = Model.feature_importances_
    print(feature_importance)
    # Umrechnen auf Prozent
    feature_importance = feature_importance*100
    # Sortieren der Feature-Importance und deren Indizes
    sorted_indices = np.argsort(feature_importance)[::-1]
    #print(sorted_indices)
    #Umbenennung von Phi Ist
    X_train = X_train.rename(columns={'phi-Ist': 'Phi-Ist'})
    # Sortiere die Features nach der Wichtigkeit
    sorted_features = np.array(X_train.columns)[sorted_indices]
    #print(sorted_features)
    sorted_importance = feature_importance[sorted_indices]
    #print(sorted_importance)
    # Ausgabe der Feature-Importance
    for i, (feature, importance) in enumerate(zip(X_train.columns[sorted_indices], sorted_importance), 1):
        print(f"{i}. {feature}: {importance}")

        # Sicherstellen, dass das Verzeichnis existiert
    os.makedirs(Sicherungsordner, exist_ok=True)

    #Farbe für die Balken (wie Header in Thesis)
    color_thesis =(0, 90/255, 169/255) # Dunkelblau der Thesis
        
    #Erstellen des Plotes mit verschiedenen Parametern
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')  
    ax.bar(range(len(sorted_importance)), sorted_importance, tick_label=sorted_features, color=color_thesis)
    ax.set_title('Wichtigkeit der Features', pad=15, fontsize=20)
    ax.set_xlabel('Features', labelpad=10, fontsize=18)
    ax.set_ylabel(f'Wichtigkeit der Features in $\\it{{\\%}}$', labelpad=10, fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
    #ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)  # Setzt das Grid in den Hintergrund

    # plt.xticks(rotation=90)

    # Iteration über die x Achsen Plots um Wert der Balken anzugeben 
    for p in ax.patches:
        height = p.get_height()
        # Wenn der Wert kleiner 0.0009 ist wird 0 angegeben
        display_value = 0 if height < 0.0009 else height
        # Hier kann über f die Anzahl an Kommastellen angegeben werden
        ax.annotate(f'{display_value:.3f}' if display_value != 0 else '0',
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16)
            
        # Problem mit Beschriftung der Balken (krzeut immer den Rand des Diagramms), deshalb wird hier der höchste Wert ermittelt und eine Obergrenze festgelegt
        max_value = sorted_importance.max()  # Höchster Balkenwert
        plt.ylim(0, max_value * 1.2)  # Vergrößern der oberen Grenze um 30% zum Maximalwert
        
    plt.tight_layout()
    # Speichern der Plots als SVG und PNG
    plt.savefig(os.path.join(Sicherungsordner, f'{model_name}_FeatureImportance_Random_11_StandardSplit_RFR.svg'), format='svg')
    plt.savefig(os.path.join(Sicherungsordner, f'{model_name}_FeatureImportance_Random_11_StandardSplit_RFR.png'), format='png')
    plt.show()

    