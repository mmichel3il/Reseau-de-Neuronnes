class PerceptronMultiCouches:
    def __init__(self, architecture, learning_rate=0.01, activation='sigmoid'):
        """
        architecture: liste des tailles de couches [input_size, hidden1, hidden2, ..., output_size]
        """
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.activation = activation
        self.couches = []
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

        # Création des couches
        for i in range(len(architecture) - 1):
            # TODO: Créer les couches successives
            # La dernière couche peut avoir une activation différente
            activation_couche = activation
            if i == len(architecture) - 2:  # Dernière couche
                activation_couche = 'sigmoid'  # ou 'softmax' pour multi-classes

            couche = CoucheNeurones(
                n_input=architecture[i],
                n_neurons=architecture[i+1], 
                activation=activation_couche,
                learning_rate=learning_rate
            )
            self.couches.append(couche)

    def forward(self, X):
        """
        Propagation avant à travers tout le réseau
        """
        current_input = X.T  # Transposer pour avoir (n_features, n_samples)
        for couche in self.couches:
            current_input = couche.forward(current_input)

        return current_input.T  # Retransposer pour avoir (n_samples, n_output)

    def backward(self, X, y_true, y_pred):
        """
        Rétropropagation à travers tout le réseau
        """
        # TODO: Calculer le gradient initial (dérivée de la fonction de coût)
        # Pour l'erreur quadratique : gradient = (y_pred - y_true)
        # TODO: Propager le gradient vers l'arrière

    def train_epoch(self, X, y):
        """
        Une époque d'entraînement
        """
        # Propagation avant
        y_pred = self.forward(X)

        # Calcul de la fonction de perte
        loss = self.compute_loss(y, y_pred)

        # Rétropropagation
        self.backward(X, y, y_pred)

        return loss, y_pred

    def compute_loss(self, y_true, y_pred):
        """
        Calcule la fonction de coût (erreur quadratique moyenne)
        """
        # TODO: Implémenter l'erreur quadratique moyenne
        return 0

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, verbose=True):
        """
        Entraîne le réseau
        """
        for epoch in range(epochs):
            # Entraînement
            loss, y_pred = self.train_epoch(X, y)
            accuracy = self.compute_accuracy(y, y_pred)

            self.history['loss'].append(loss)
            self.history['accuracy'].append(accuracy)

            # Validation si données fournies
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_loss = self.compute_loss(y_val, y_val_pred)
                val_accuracy = self.compute_accuracy(y_val, y_val_pred)

                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)

            if verbose and epoch % 10 == 0:
                print(f"Époque {epoch:3d} - Loss: {loss:.4f} - Acc: {accuracy:.4f}")

    def predict(self, X):
        """
        Prédiction sur de nouvelles données
        """
        return self.forward(X)

    def compute_accuracy(self, y_true, y_pred):
        """
        Calcule l'accuracy pour la classification binaire
        """
        # TODO: Implémenter le calcul d'accuracy
        # Pour la classification binaire : seuil à 0.5
        predictions = (y_pred > 0.5).astype(int)
        return np.mean(predictions.flatten() == y_true.flatten())