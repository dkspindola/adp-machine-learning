from src.funcions_konventionelle_Modelle.Hyperparametertuning_Konventionell import random_forest_hyperparameter_search
# Beispielaufruf der Funktionen
# unnötige Aufrufe können auskommentiert werden
if __name__ == "__main__":
    #Hyperparametrisierung
    from lib.Load_Data_for_Modelling import Get_data
    data = Get_data(0, 1800, 0)
    X_train, X_test, Y_train, Y_test = Split_Scaling(data, size=0.1, Train_Test_Split=2, Datengröße=1800, random=5)
    save_dir_hyper = r'C:\Users\corvi\OneDrive - stud.tu-darmstadt.de\Desktop\Masterthesis\14_Modelle_Hyperparameter\Konventionell'
    #Hyperparametertuning mit Random Forest der konventionellen Modelle
    random_forest_hyperparameter_search(data, save_path=save_dir_hyper)
    #Hyperparameter validieren der konventionellen Modelle für Random Forrest
    validate_best_model_random_forest(rf_random, X_train, X_test, Y_train, Y_test)
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
    from lib.Load_Data_for_Modelling import Get_data
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

