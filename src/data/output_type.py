from enum import Enum

class OutputTarget(Enum):
    """
    Klasse mit den einzelnen Bezeichnungne der Outputs der Modelle.
    Dann ist es wichtig, dass die Objective des tunings oder des Fits stimmen. 
    Dafür unten die getter, die deise dem Modelloutput zuweisen. """
    VERSTELLWEG_X = 'Verstellweg_X'
    VERSTELLWEG_Y = 'Verstellweg_Y'
    VERSTELLWEG_PHI = 'Verstellweg_Phi'

    def get_objective(self):
        # Gibt z.B. "val_Verstellweg_X_mae" zurück
        return f"val_{self.value}_mae"

    def get_loss_metric_dict(self):
        return {self.value: ['mae']}

    def get_loss_dict(self):
        return {self.value: 'mean_absolute_error'}

    def get_output_name(self):
        return self.value

    def get_monitor_mode(self):
        return 'min'  # da wir MAE verwenden
    
    def get_index(self) -> int:
        """
        Gibt den Index dieses Outputs basierend auf der Reihenfolge zurück.
        ACHTUNG: DIe Reihnfolge der Werte in den Y-Daten muss damit übereinstimmen.
        """
        return list(OutputTarget).index(self)