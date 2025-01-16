# Beispielaufruf der Funktionen
# unnötige Aufrufe können auskommentiert werden
if __name__ == "__main__":
    #Hyperparametrisierung
    from Load_Data_for_Modelling import Get_data
    data = Get_data(0, 1800, 0)
    X_train, X_test, Y_train, Y_test = Split_Scaling(data, size=0.1, Train_Test_Split=2, Datengröße=1800, random=5)
    save_dir_hyper = r'C:\Users\corvi\OneDrive - stud.tu-darmstadt.de\Desktop\Masterthesis\14_Modelle_Hyperparameter\Konventionell'
    #Hyperparametertuning mit Random Forest der konventionellen Modelle
    random_forest_hyperparameter_search(data, save_path=save_dir_hyper)
    #Hyperparameter validieren der konventionellen Modelle für Random Forrest
    validate_best_model_random_forest(rf_random, X_train, X_test, Y_train, Y_test):
    #Hyperparametertuning mit K-Neighbour der konventionellen Modelle
    knr_hyperparameter_tuning(data, save_path=save_dir_hyper)
    #Hyperparameter validieren der konventionellen Modelle für K-Neighbours
    validate_knr_model(KNR_opt, X_train, Y_train, X_test, Y_test)
    #Hyperparametertuning + Valdiierung mit Decision Tree
    decision_tree_hyperparameter_search(data)
    #Hyperparametrisierung + Validierung mit SVR
    svr_hyperparameter_search(data)



    #Modellierung
    from Splitting_Scaling_Function_ValData import Split_Scaling
    from Load_Data_for_Modelling import Get_data
    data = Get_data(0, 1800, 0)
    #Modellierung und Ausgabe
    results_ML = ML(data, ML_Model=4, x_start=0, x_end=1800, size=0.2, Train_Test_Split=2, Datengröße=1800, random=42)
    Y_pred_train, Y_pred_test, MAE_train, MAE_test, M2E_train, M2E_test, model_name, model, df_Fehler = results_ML
    
    #Validierung der Modelle
    modelle = [1, 2, 4, 5]  # Liste von Modellnummern
    random_seed_list = random.sample(range(101), 10) #zufällige Seeds
    result_ML_validiert = Validierung_ML(data, modelle, random_seed_list)
    MAE_StandardSplit_df, M2E_StandardSplit_df, MAE_BlechSplit_df, M2E_BlechSplit_df, Df_Fehler_append_Standardsplit, Df_Fehler_append_Blechsplit = result_ML_validiert

    #Errorconlusion und Analyse
    folder_path = r'C:\Users\corvi\OneDrive - stud.tu-darmstadt.de\Desktop\Masterthesis\13_ExcelvonDaten_Code\Konventionelle Modelle'
    error_conclusion(MAE_BlechSplit_df, MAE_StandardSplit_df, M2E_BlechSplit_df, M2E_StandardSplit_df, Df_Fehler_append_Standardsplit, Df_Fehler_append_Blechsplit, folder_path)
    threshold = 100
    results = error_analysis(folder_path, threshold)

    #Feature Importance
    output_folder = r'C:\\Users\\corvi\\OneDrive - stud.tu-darmstadt.de\\Desktop\\Masterthesis\\15_Plots\\Konventionelle Modelle\\Feature_Importance-RFR'
    feature_importance(model, X_train, model_name, output_folder)

    #______________________________________________________________________________________________________________________

    #Neuronale Netze
    #Imports
    from Fensterung_Scaling_DeepLearning import windowing_scale
    from Load_Data_for_Modelling import Get_data
    data = Get_data(0, 1800, 0,1)
    X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, Angepasste_Blechnummern_test = windowing_scale(Data, Validation_data=1, random=7, Train_Test_Split=2, window_size=10)
    import numpy as np

    # Bevor die Daten für die Hyperparametersuche verwendet werden müssen die Labels "gesqueezed" werden für ein eindimensionales Array
    Y_train = np.squeeze(Y_train)
    Y_test = np.squeeze(Y_test)
    Y_val =np.squeeze(Y_val)
    Y_train_scaled = np.squeeze(Y_train_scaled)
    Y_val_scaled = np.squeeze(Y_val_scaled)

    tuner_dir = './'my_dir'r'
    save_dir = r'C:\Users\corvi\OneDrive - stud.tu-darmstadt.de\Desktop\Masterthesis\14_Modelle_Hyperparameter'
    os.makedirs(save_dir, exist_ok=True)

    #Bayesian Hyperparameter Search
    best_hyperparams, best_trained_model = bayesian_hyperparameter_search(
        X_train_scaled, Y_train, X_val_scaled, Y_val, X_test_scaled, Y_test, tuner_dir, save_dir)
    #Modell bauen und Traineren
    mae_X, mae_Y, mae_Phi, df_Fehler = bestes_model(X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels)
    #Dieses erzeugtes Modell validieren (ohne Interpolierte Daten)
    MAE_Standardsplit, MAE_Blechsplit, Fehler_Standardsplit_df, Fehler_Blechsplit_df = validiere_modelle_oI(data)
    #Dieses erzeugtes Modell validieren (mit Interpolierte Daten)
    MAE_Standardsplit, Fehler_Standardsplit_df, MAE_Blechsplit, Fehler_Blechsplit_df = validiere_modelle_mI(data)
    #abschließende Statistische Berechnungen
    Ordner = r'C:\Users\corvi\OneDrive - stud.tu-darmstadt.de\Desktop\Masterthesis\13_ExcelvonDaten_Code\DeepLearning\CNN'
    statische_Berechnungen(MAE_Standardsplit, MAE_Blechsplit, Ordner)


    #______________________________________________________________________________________________________________________

    #CNN mit Augmentation Mode (erweiterter Datenraum durch interpolation)
    from Load_Data_for_Modelling_Function import Data_for_Model
    from Load_Data_for_Modelling_Interpolation import Interpolation
    from Fensterung_Scaling_CNN_ValData_Interpolation_Test import Fensterung_Scale
    data = Data_for_Model(0,1800)
    df_Int, Interpoliertes_df = Interpolation(0,1800,0)
   
    # Falls Validations Daten benötigt werden ohne Interpolation 

    X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, Angepasste_Blechnummern_test = windowing_scale(data, Validation_data=1, random=8, Train_Test_Split=2, window_size=10)
    # Falls Validations Daten benötigt und interpoliert wird
    X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled, Y_train, Y_val, Y_test, scalers_features, scaler_labels, X_test_scaled_int, Y_test_interpolation, Blechnummern_Test_Int = windowing_scale(df_Int, interpoliertesdf=Interpoliertes_df, Validation_data=1, random=8, Train_Test_Split=2, window_size=10, Interpolation=1)
    # Falls keine Validationsdaten benötigt werden 
    #X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled, Y_train, Y_test, scalers_features, scaler_labels = Fensterung_Scale(Validation_data=0, random=42)

    #Calculations
    cnn_df, transformer_df = calculate_for_interpolation(Random_split=11, num_loops=4)
    output_folder = r'C:\Users\corvi\OneDrive - stud.tu-darmstadt.de\Desktop\Masterthesis\13_ExcelvonDaten_Code\Interpolation'
    filename = 'Vergleich_CNN_Tranformer_Random11_Intfaktor0_3_x.csv''
    save_calculations(cnn_df, transformer_df, output_folder, filename)
