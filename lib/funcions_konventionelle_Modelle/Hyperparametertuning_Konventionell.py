
# Import für def random_forest_hyperparameter_search
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
import json
import joblib

# Import für Validierungsfunktionen (Random FOrrst und KNR)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import für def knr_hyperparameter_search
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from lib.Load_Data_for_Modelling import Get_data
import os
import json
import joblib
from sklearn.metrics import make_scorer, mean_absolute_error
from lib.Splitting_Scaling_Function import Split_Scaling

# Imprt für DesicionTree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#Import für SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np


# Random Forest Hyperparametertuning mittels RandomizedSearchCV
def random_forest_hyperparameter_search(data, 
                                        size=0.2, 
                                        train_test_split_method=2, 
                                        datengroesse=1800, 
                                        random_seed=42, 
                                        save_path='.', 
                                        n_iter=50, 
                                        cv=3,
                                        verbose=2,
                                        n_jobs=-1):

    '''Parameter erklärung:
        data: Daten, die durch Get_data bereitgestellt werden.
        size: Anteil der Testdaten.
        train_test_split_method: Methode zur Durchführung des Splits.
        datengroesse: Größe des Datensatzes.
        random_seed: Seed für Reproduzierbarkeit.
        save_path: Verzeichnis, in dem die Ergebnisse gespeichert werden.
        n_iter: Anzahl der getesteten Hyperparameter-Kombinationen.
        cv: Anzahl der Folds für Cross-Validation.
        verbose und n_jobs: Auch irgendwelche Parameter (Erklärung kommt noch)

        Sonstige Parameter (scoring, random_state, estimator, param_distribution) können hier hinzugefügt werden.
    '''
    # Daten werden geladen mittels "data" in main
    # Daten aufteilen
    X_train, X_test, Y_train, Y_test = Split_Scaling(data, 
                                                     size=size, 
                                                     Train_Test_Split=train_test_split_method, 
                                                     Datengröße=datengroesse, 
                                                     random=random_seed)
    # Hyperparameter-Raum festlegen:
    
    # Anzahl an Bäume
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Anzahl an Features bei jedem Split
    max_features = ['log2', 'sqrt', 10, 0.5]
    # Maximale tiefe der Bäume
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum von Samples einen Baum zu splitten
    min_samples_split = [2, 5, 10]
    # Minimum um ein Blattknoten zu splitten
    min_samples_leaf = [1, 2, 3, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Erzeuge das Random grid mit den festgelegten Hyperparametern
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    print(random_grid)
    
    # Lege den RFR fest
    rf_op=RandomForestRegressor()

    # Random search of parameters, 3 Folds pro Suche
    # Suche 50 Kombinationen und validiere jede 3 mal 
    rf_random = RandomizedSearchCV(estimator = rf_op, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)

    # Modell fitten um Suche auszuführen
    rf_random.fit(X_train, Y_train)

    # Printe die besten Hyperparameter
    print("Beste Parameter:", rf_random.best_params_)

    # Speichern der besten Parameter
    best_params = rf_random.best_params_
    Pfad = r'C:\Users\corvi\OneDrive - stud.tu-darmstadt.de\Desktop\Masterthesis\14_Modelle_Hyperparameter\Konventionell'
    hyperparameters_pfad = os.path.join(Pfad, 'best_hyperparameters_RandomForest.json')
    with open(hyperparameters_pfad, 'w') as json_file:
        json.dump(best_params, json_file)

    # Speichern des besten Modells
    model_pfad = os.path.join(Pfad, 'best_random_forest_model.pkl')
    joblib.dump(rf_random.best_estimator_, model_pfad)


#Validieren des besten Modells für Random Forrest 
def validate_best_model_random_forest(rf_random, X_train, X_test, Y_train, Y_test):
    '''Parameter erklärung:
        rf_random: Das beste Modell, welches durch die Funktion random_forest_hyperparameter_search gefunden wurde.
        X_train, X_test, Y_train, Y_test: Die Daten, die durch Get_data bereitgestellt werden.
    '''
    # Beste Parameter und bestes Modell anzeigen
    #print("Beste Parameter:", rf_random.best_params_)

    # Predicte die Variablen 
    print(f'\tCalculate predictions', end='\r')        
    Y_pred_best_train = rf_random.predict(X_train)
    Y_pred_best_test = rf_random.predict(X_test)
    print('✔\n')

    # MAEs berechnen für die predicteten Values mit den besten Hyperparameter
    print(f'\tCalculate mean absolute error', end='\r')        
    MAE_best_train = mean_absolute_error(Y_train, Y_pred_best_train, multioutput='raw_values')
    MAE_best_test = mean_absolute_error(Y_test, Y_pred_best_test, multioutput='raw_values')
    print('✔')
     #Print des MAEs
    print(f'\tMAE(training):\t{MAE_best_train}')
    print(f'\tMAE(test)):\t{MAE_best_test}')
    print('')

    # R2 Score
    print(f'\tCalculate R2 score', end='\r')        
    r2__best_train = r2_score(Y_train, Y_pred_best_train, multioutput='raw_values')
    r2_best_test = r2_score(Y_test, Y_pred_best_test, multioutput='raw_values')
    print('✔')
    #Print des R2
    print(f'\tR2(training):\t{r2__best_train}')
    print(f'\tR2(test):\t{r2_best_test}')
    print('')

    



#Hyperparametertuning mittels K-Neighbours
def knr_hyperparameter_search(data, 
                              size=0.2, 
                              train_test_split_method=2, 
                              datengroesse=1800, 
                              random_seed=42, 
                              validation_data=0, 
                              save_path='.', 
                              n_iter=50, 
                              cv=3, 
                              verbose=2, 
                              n_jobs=-1):

    """
    Parameters:
        data: Daten, die durch Get_data bereitgestellt werden.
        size: Anteil der Testdaten.
        train_test_split_method: Methode zur Durchführung des Splits.
        datengroesse: Größe des Datensatzes.
        random_seed: Seed für Reproduzierbarkeit.
        validation_data: Anteil der Validierungsdaten (optional).
        save_path: Verzeichnis, in dem die Ergebnisse gespeichert werden.
        n_iter: Anzahl der getesteten Hyperparameter-Kombinationen.
        cv: Anzahl der Folds für Cross-Validation.
        verbose: Grad der Ausgabe bei der Hyperparameter-Suche (z. B. Fortschritt).
        n_jobs: Anzahl der parallelen Jobs (-1 für alle verfügbaren Kerne).
    """

    # Reinladen der Daten über Inputvariable data 
    # Aufteilen der Daten
    X_train, X_test, Y_train, Y_test = Split_Scaling(
    data,
    size=size,
    Train_Test_Split=train_test_split_method,
    Datengröße=datengroesse,
    random=random_seed,
    Validation_Data=validation_data
    )

    # Definition des Hyperparameterraums
    random_grid = {'n_neighbors': [3, 5, 7,9,11,13,15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'metric' :['minkowski','euclidean','manhattan']
        }

    #Initialisere KNR
    KNR = KNeighborsRegressor()

    # Definieren Sie die Scoring-Funktion
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Durchführung der Hyperparametersuche (Konfiguration und Durchführung der SUche)
    KNR_opt = RandomizedSearchCV(
    estimator=KNR,
    param_distributions=random_grid,
    n_iter=n_iter,
    cv=cv,
    scoring=mae_scorer,
    verbose=verbose,
    n_jobs=n_jobs,
    random_state=random_seed
    )

    KNR_opt.fit(X_train, Y_train)

    # Beste Parameter und bestes Modell anzeigen
    print("Beste Parameter:", KNR_opt.best_params_)

    # Speichern der besten Parameter
    best_params = KNR_opt.best_params_
    hyperparameters_pfad = os.path.join(save_path, 'best_hyperparameters_KNR_30Trials.json')
    with open(hyperparameters_pfad, 'w') as json_file:
        json.dump(best_params, json_file)

    # Speichern des besten Modells
    model_pfad = os.path.join(save_path, 'best_KNR_model.pkl')
    joblib.dump(KNR_opt.best_params_, model_pfad)


# Validieren des besten Modells des KNRs
def validate_knr_model(KNR_opt, X_train, Y_train, X_test, Y_test):
    
    '''    Parameters:
        KNR_opt: Das trainierte KNeighborsRegressor-Modell.
        X_train: Trainingsdaten (Features).
        Y_train: Trainingsdaten (Labels).
        X_test: Testdaten (Features).
        Y_test: Testdaten (Labels).'''
    # Predicte die Values
    Y_pred_best_train = KNR_opt.predict(X_train)
    Y_pred_best_test = KNR_opt.predict(X_test)

    # MAE der Vorhersagen
    MAE_best_train = mean_absolute_error(Y_train, Y_pred_best_train, multioutput='raw_values')
    MAE_best_test = mean_absolute_error(Y_test, Y_pred_best_test, multioutput='raw_values')

    # R2 Score
    r2__best_train = r2_score(Y_train, Y_pred_best_train, multioutput='raw_values')
    r2_best_test = r2_score(Y_test, Y_pred_best_test, multioutput='raw_values')

    #Printe MAEs
    print(f'MAE für die Trainingsdaten des best fits: {MAE_best_train}')
    print(f'MAE für die Testdaten des best fits: {MAE_best_test}')

    # Printe R2 Score
    print(f'R2-Score für die Trainingsdaten des best fits: {r2__best_train}')
    print(f'R2-Score für die Testdaten des best fits: {r2_best_test}')

# Decision Tree Regressor Hyperparametersuche + Validierung
def decision_tree_hyperparameter_search(data, size=0.2, train_test_split=2, datengroesse=1800, random=42):
    """
    Führt eine Hyperparametersuche und Validierung für einen DecisionTreeRegressor durch.

    Parameters:
        data: Datensatz, der durch Get_data bereitgestellt wird.
        size: Anteil der Testdaten (default: 0.2).
        train_test_split: Methode für den Split (default: 2).
        datengroesse: Gesamtgröße des Datensatzes (default: 1800).
        random: Seed für Reproduzierbarkeit (default: 42).

    """
    # Daten werden über die Variable data geladen
    # Datenaufbereitung
    X_train, X_test, Y_train, Y_test = Split_Scaling(
        data, size=size, Train_Test_Split=train_test_split, Datengröße=datengroesse, random=random
    )
    # Definition des Modells
    model = DecisionTreeRegressor()

    # Definition des Parameterraums
    param_grid = {
        'criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10]
    }

    # Durchführung der Rastersuche mit MAE als Bewertungsmetrik
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, Y_train)

    # Beste Parameter und bestes Modell anzeigen
    print("Beste Parameter:", grid_search.best_params_)

    Y_pred_best_train = KNR_opt.predict(X_train)
    Y_pred_best_test = KNR_opt.predict(X_test)

    MAE_best_train = mean_absolute_error(Y_train, Y_pred_best_train, multioutput='raw_values')
    MAE_best_test = mean_absolute_error(Y_test, Y_pred_best_test, multioutput='raw_values')

    r2__best_train = r2_score(Y_train, Y_pred_best_train, multioutput='raw_values')
    r2_best_test = r2_score(Y_test, Y_pred_best_test, multioutput='raw_values')

    print(f'MAE für die Trainingsdaten des best fits: {MAE_best_train}')
    print(f'MAE für die Testdaten des best fits: {MAE_best_test}')

    print(f'R2-Score für die Trainingsdaten des best fits: {r2__best_train}')
    print(f'R2-Score für die Testdaten des best fits: {r2_best_test}')


#SVR Hyperparametertuning mittels RandomizedSearchCV + Validierung
def svr_hyperparameter_search(data, size=0.2, train_test_split=2, datengroesse=1800, n_iter=50, cv=3, verbose=2, random_state=42):
    """
    Parameters:
        data: Datensatz, der durch Get_data bereitgestellt wird.
        size: Anteil der Testdaten (default: 0.2).
        train_test_split: Methode für den Split (default: 2).
        datengroesse: Gesamtgröße des Datensatzes (default: 1800).
        random: Seed für Reproduzierbarkeit (default: 42).
    """
    #Daten werden über die Variable data reingeladen
    # Daten aufteilen
    X_train, X_test, Y_train, Y_test = Split_Scaling(
        data, size=size, Train_Test_Split=train_test_split, Datengröße=datengroesse, random=random
    )
    # Cross validation grid search (beste Parameters) 
    c_range = np.logspace(-0, 4, 8)
    gamma_range = np.logspace(-4, 0, 8)
    epsilon = [0.1,0.2,0.3]
    kernel= ['rbf','linear','sigmoid']

    # Erzeuge das Random Grid
    random_grid = {'C': c_range,
                'gamma': gamma_range,
                'kernel': kernel,
                'epsilon': epsilon
                }

    print(random_grid)
    print(Y_train['Y_Opt-Y_ist'])
    # Initalisiere Modelle
    svr = SVR()

    # SVR muss als MultiOutputRegressor definiert werden
    SVR_op = MultiOutputRegressor(svr)

    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    SVR_random = RandomizedSearchCV(
        svr,
        param_distributions=random_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=verbose,
        random_state=random_state
    )
    # Fit the random search model
    SVR_random.fit(X_train, Y_train)

    # Validiere das Modell mit den Testdaten und dem entsprechenden MAE und R2 Score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    print(SVR_random.best_params_)

    Y_pred_best_train_SVR = SVR_random.predict(X_train)
    Y_pred_best_test_SVR = SVR_random.predict(X_test)

    MAE_best_train = mean_absolute_error(Y_train['X_opt-X-Ist'], Y_pred_best_train_SVR, multioutput='raw_values')
    MAE_best_test = mean_absolute_error(Y_test['X_opt-X-Ist'], Y_pred_best_test_SVR, multioutput='raw_values')

    r2__best_train = r2_score(Y_train['Y_Opt-Y_ist'], Y_pred_best_train_SVR, multioutput='raw_values')
    r2_best_test = r2_score(Y_test['Y_Opt-Y_ist'], Y_pred_best_test_SVR, multioutput='raw_values')

    print(f'MAE für die Trainingsdaten des best fits: {MAE_best_train}')
    print(f'MAE für die Testdaten des best fits: {MAE_best_test}')

    print(f'R2-Score für die Trainingsdaten des best fits: {r2__best_train}')
    print(f'R2-Score für die Testdaten des best fits: {r2_best_test}')








