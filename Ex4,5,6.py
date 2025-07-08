#exercice4
import numpy as np
import matplotlib.pyplot as plt

class PerceptronSimple:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y, max_epochs=100):
        """
        Entraîne le perceptron
        X: matrice des entrées (n_samples, n_features)
        y: vecteur des sorties désirées (n_samples,)
        """
        # Initialisation les poids et le biais
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
                # TODO: Implémenter l'algorithme d'apprentissage
                # Possible d'optimiser d'aventage numpy
                if error != 0:
                    self.weights += self.learning_rate * error * x
                    self.bias += self.learning_rate * error
                    countError += 1
                    
            if countError == 0 :
                print(f"Convergence atteinte après {e} époques.")
                break
              
                
    def predict(self, X):
        """Prédit les sorties pour les entrées X"""
        y_pred = np.zeros(X.shape[0])

        for b in range(X.shape[0]):
            # TODO: Calculer les prédictions
            x = X[b] # (n_features,)
            z = np.dot(x, self.weights) + self.bias
            y_pred[b] = 1 if z>=0 else 0

        return y_pred

    def score(self, X, y):
        """Calcule l'accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
        
        
        
      
      
X = np.array([
 [0, 0],
 [0, 1],
 [1, 0],
 [1, 1]
])

y = np.array([0, 0, 0, 1])  # fonction AND

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 0, 0, 1])

# Créer et entraîner le perceptron
p = PerceptronSimple(learning_rate=0.1)
p.fit(X_xor, y_xor, max_epochs=100)

# Tracer la frontière de décision
plt.figure(figsize=(8,6))
for i in range(len(X)):
    if y[i] == -1:
        plt.scatter(X[i,0], X[i,1], color='red', label='Classe -1' if i==0 else "")
    else:
        plt.scatter(X[i,0], X[i,1], color='blue', label='Classe 1' if i==1 else "")

# Équation de la droite
x_vals = np.linspace(-0.1, 1.1, 100)
w1, w2 = p.weights
b = p.bias

if w2 != 0:
    y_vals = -(w1 / w2) * x_vals - (b / w2)
    plt.plot(x_vals, y_vals, 'k--', label="Frontière de décision")

plt.title("Frontière de séparation du perceptron")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.legend()
plt.grid(True)
plt.show()
# Tester le modèle
print("Poids appris :", p.weights)
print("Biais appris :", p.bias)
print("Prédictions :", p.predict(X))
print("Vérités     :", y)
print("Précision   :", p.score(X, y))
