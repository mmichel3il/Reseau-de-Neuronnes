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
        # Supprimer ou commenter la ligne suivante pour obtenir des données différentes à chaque exécution :
        # np.random.seed(42)
        n = n_points // 2
        classe_pos = np.random.randn(n, 2) * noise + np.array([2, 2])
        classe_neg = np.random.randn(n, 2) * noise + np.array([-2, -2])
        X = np.vstack((classe_pos, classe_neg))
        y = np.hstack((np.ones(n), np.zeros(n)))
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

    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def analyser_convergence(X, y, learning_rates=[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0]):
    """
    Analyse la convergence pour différents taux d'apprentissage
    """
    plt.figure(figsize=(12, 8))

    for i, lr in enumerate(learning_rates):
        np.random.seed(i + 123)  # Seed différente pour chaque taux
        p = PerceptronSimple(learning_rate=lr)
        p.weights = np.random.randn(X.shape[1])  # Initialisation unique
        p.bias = 0.0
        errors_per_epoch = []

        for epoch in range(100):
            errors = 0
            for xi, yi in zip(X, y):
                z = np.dot(xi, p.weights) + p.bias
                y_pred = 1 if z >= 0 else 0
                error = yi - y_pred

                if error != 0:
                    p.weights += lr * error * xi
                    p.bias += lr * error
                    errors += 1

            errors_per_epoch.append(errors)
            if errors == 0:
                break

        plt.plot(errors_per_epoch, label=f"η = {lr}")
        print(f"Taux d'apprentissage {lr} → convergence en {len(errors_per_epoch)} époques")

    plt.xlabel('Époque')
    plt.ylabel("Nombre d'erreurs")
    plt.title("Convergence pour différents taux d'apprentissage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# === UTILISATION ===

# Générer les données (n'oublie pas : seed désactivée pour données variables)
X, y = PerceptronSimple.generer_donnees_separables(n_points=100, noise=2.0)

# Entraîner une fois pour visualiser la frontière
p = PerceptronSimple(learning_rate=0.01)
p.fit(X, y)
visualiser_donnees(X, y, w=p.weights, b=p.bias, title="Données et frontière de décision")
print("Poids appris :", p.weights)
print("Biais appris :", p.bias)
print("Précision :", p.score(X, y))

# Exercice 8 : Analyse de la convergence
analyser_convergence(X, y)