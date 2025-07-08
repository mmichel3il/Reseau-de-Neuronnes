import numpy as np
from tqdm import tqdm
from PerceptronEx4 import PerceptronSimple

class PerceptronMultiClasse:
    def init(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.perceptrons = {}
        self.classes = None

    def fit(self, X, y, max_epochs=100):
        """
        Entraîne un perceptron par classe (stratégie un-contre-tous)
        """
        self.classes = np.unique(y)

        for classe in tqdm(self.classes, desc="Entraînement des perceptrons"):
            # TODO: Créer un problème binaire pour cette classe
            y_pred = np.where(classe < 0, 0, 1)

            #Transformer y en problème binaire : classe courante vs toutes les autres
            y_binary = np.where(y == classe, 1, -1)

            # TODO: Entraîner un perceptron pour ce problème binaire
            perceptron = PerceptronSimple(learning_rate=self.learning_rate)
            perceptron.fit(X, y_binary, max_epochs)

            # Stocker le perceptron entraîné
            self.perceptrons[classe] = perceptron

    def predict(self, X):
        """Prédit en utilisant le vote des perceptrons"""
        if not self.perceptrons:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")

        # TODO: Calculer les scores de tous les perceptrons
        scores = np.zeros((X.shape[0], len(self.classes)))

    def predict_proba(self, X):
        """Retourne les scores de confiance pour chaque classe"""
        if not self.perceptrons:
            raise ValueError("Le modèle n'a pas été entraîné.")

        scores = np.zeros((X.shape[0], len(self.classes)))

        for i, classe in enumerate(self.classes):
            perceptron = self.perceptrons[classe]
            raw_scores = X.dot(perceptron.weights) + perceptron.bias
            scores[:, i] = raw_scores

        return scores

