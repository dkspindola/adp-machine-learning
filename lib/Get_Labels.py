import pandas as pd
import os
# Excel Datei Labels muss im entsprechenden Ordner liegen um Funktion zu durchlaufen
# Falls die Excel Datei mit den umgerechneten Labels erzeugt und gespecihert werden soll, muss der Funktion gleich 1 übergeben werden Get_Label(Speichern=1)

def Get_Label(Speichern=0):
    
    # Hier den Ordner angeben in dem die Labels Datei liegt
    Ordner = r'C:\Users\corvi\OneDrive - stud.tu-darmstadt.de\Desktop\Masterthesis\10_Daten_Versuche' 
    Datei = 'Labels.csv' #Name der Excel Datei
    Datei_Pfad = os.path.join(Ordner,Datei)
    Labels = pd.read_csv(Datei_Pfad, sep=';') #Einlesen der Excel
    Labels = Labels.iloc[:,0:8] #Abschneiden der Beschreibung in der Excel (siehe Excel Labels)

    #print(Labels)
    # Umrechnen der Labels von Werten an dem Richtapparat zu mm bzw. ° 
    Labels['X_opt-X-Ist'] = Labels['X_opt-X-Ist']*(1/10)
    Labels['Y_Opt-Y_ist'] = Labels['Y_Opt-Y_ist']*(1/10)
    Labels['phi_Opt-phi_ist'] = Labels['phi_Opt-phi_ist']*(1/10)*(1.75)

    # Ist-Positionen umrechnen: Ausgangsposition ist immer X=90, Y=99700, Phi=1, Werte müssen vorher entsprechend umgerechnet werden weil mit phi=1-41 und y=1 etc nicht gerechnet werden kann
    Labels['phi-Ist']= Labels['phi-Ist'].replace([1,11,21,41],[10001,10011,10021,10041])
    Labels['Y-Ist']=Labels['Y-Ist'].replace(0,100000)

    # Aktuelle Positionen auf Basis des Ausgangszustands berechnen 
    Labels['X-Ist']=(Labels['X-Ist']-90)*1/10
    Labels['Y-Ist']=(99700-Labels['Y-Ist'])*(1/10)
    Labels['phi-Ist']=(Labels['phi-Ist']-10001)*(1/10)*1.75

    #print(Labels)

    # Falls Excel mit den umgerechneten labels gespeichert werden soll
    if Speichern == 1:
        
    # Wird benötigt um die CSV richtig zu erstellen, ansonsten ersetzt er die 3.5 zu 3.Mai #Für die weitere Verwendung muss es auskommentiert werden
    # Ersetze Punkt durch Komma
        for Column in Labels.columns:
            Labels[Column] = Labels[Column].astype(str).str.replace('.', ',')

        #print(type(Labels['phi_Opt-phi_ist'].iat[2]))
        #Labels.to_csv(f'{Ordner}\Labels_Angepasst_fuerModellierung.csv', index=False, sep=';') # Umgerechnete Labels werden in eine neue Excel geladen, am gleichen Ort an dem die vorherige Datei Labels gespeichert ist
    
    return Labels