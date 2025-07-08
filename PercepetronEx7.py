import numpy as np
import matplotlib.pyplot as plt

class PerceptronSimple:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y, max_epochs=100):
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0.0

        for e in range(max_epochs):
            countError = 0
            for b in range(X.shape[0]):
                x = X[b]
                y_true = y[b]
                z = np.dot(x, self.weights) + self.bias
                y_pred = 1 if z >= 0 else 0
                error = y_true - y_pred

                if error != 0:
                    self.weights += self.learning_rate * error * x
                    self.bias += self.learning_rate * error
                    countError += 1

            if countError == 0:
                print(f"Convergence atteinte après {e} époques.")
                break

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for b in range(X.shape[0]):
            x = X[b]
            z = np.dot(x, self.weights) + self.bias
            y_pred[b] = 1 if z >= 0 else 0
        return y_pred

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    @staticmethod
    def generer_donnees_separables(n_points=100, noise=0.1):
        np.random.seed(42)
        n = n_points // 2
        classe_pos = np.random.randn(n, 2) * noise + np.array([2, 2])
        classe_neg = np.random.randn(n, 2) * noise + np.array([-2, -2])
        X = np.vstack((classe_pos, classe_neg))
        y = np.hstack((np.ones(n), np.zeros(n)))  # étiquettes 1 et 0
        return X, y

def visualiser_donnees(X, y, w=None, b=None, title="Données"):
    plt.figure(figsize=(8, 6))
    mask_pos = (y == 1)
    plt.scatter(X[mask_pos, 0], X[mask_pos, 1], c='blue', marker='+', s=100, label='Classe +1')
    plt.scatter(X[~mask_pos, 0], X[~mask_pos, 1], c='red', marker='*', s=100, label='Classe -1')

    if w is not None and b is not None:
        x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
        if w[1] != 0:
            y_vals = -(w[0] * x_vals + b) / w[1]
            plt.plot(x_vals, y_vals, 'k--', label='Frontière de décision')
        else:
            x_intersect = -b / w[0]
            plt.axvline(x=x_intersect, color='k', linestyle='--', label='Frontière de décision')

def analyser_convergence(X, y, learning_rates=[0.001, 0.01, 0.1, 1.0]):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    for lr in learning_rates:
        p = PerceptronSimple(learning_rate=lr)
        errors = []

        for epoch in range(100):
            err = 0
            for xi, yi in zip(X, y):
                z = np.dot(xi, p.weights) + p.bias if p.weights is not None else 0
                y_pred = 1 if z >= 0 else 0
                error = yi - y_pred
                if error != 0:
                    if p.weights is None:
                        p.weights = np.random.randn(X.shape[1])
                        p.bias = 0.0
                    p.weights += lr * error * xi
                    p.bias += lr * error
                    err += 1
            errors.append(err)
            if err == 0:
                break

        plt.plot(errors, label=f"η = {lr}")

    plt.xlabel("Époque")
    plt.ylabel("Nombre d’erreurs")
    plt.title("Convergence pour différents taux d’apprentissage")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    analyser_convergence(X, y)

 #   plt.xlabel('x₁')
 #  plt.ylabel('x₂')
 #   plt.legend()
 #   plt.title(title)
 #   plt.grid(True, alpha=0.3)
 #   plt.show()

# === UTILISATION ===

# Générer les données linéairement séparables
X, y = PerceptronSimple.generer_donnees_separables(n_points=100, noise=0.2)

# Créer et entraîner le perceptron
p = PerceptronSimple(learning_rate=0.1)
p.fit(X, y, max_epochs=100)

# Visualisation
visualiser_donnees(X, y, w=p.weights, b=p.bias, title="Données et frontière de décision")

# Évaluation
print("Poids appris :", p.weights)
print("Biais appris :", p.bias)
print("Précision :", p.score(X, y))