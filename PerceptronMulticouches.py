import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


#On utilise la fonction sigmoïde plutôt que Heaviside parce qu’elle est dérivable partout, ce qui permet au réseau de calculer les gradients et donc d’apprendre efficacement
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class PerceptronXOR:
    def __init__(self):
        # les couche cachées A3, A4 : on a 2 neurones, chacun reçoit A1 et A2
        self.W_hidden = np.array([
            [20, 20],   # Neurone A3
            [-20, -20]  # Neurone A4
        ])
        self.b_hidden = np.array([
            [-10],  # Biais pour A3
            [30]    # Biais pour A4
        ])

        # la couche de sortie A5 : on 1 neurone,  et reçoit A3 et A4
        self.W_A5 = np.array([[20, 20]])  # Neurone A5
        self.b_A5 = np.array([[-30]])

    def forward(self, X):
        """
        elle transforme les entrées, calcule les activations intermédiaires et produit la sortie
        """
        A1A2 = X.T   #on transpose la matrice X pour avoir (2,4) au lieu de (4,2) et faire le calcule
        Z_hidden = np.dot(self.W_hidden, A1A2) + self.b_hidden      
        A3A4 = sigmoid(Z_hidden)                                   
        Z_output = np.dot(self.W_A5, A3A4) + self.b_A5    
        A5 = sigmoid(Z_output)
        return A5.T  # Retour au format (n_samples, 1)


X = np.array([
    [0, 0],  
    [0, 1], 
    [1, 0],  
    [1, 1]   
])
y_true = np.array([0, 1, 1, 0])


mlp = PerceptronXOR()
y_pred = mlp.forward(X)

print("Entrées (A1, A2) :\n", X)
print("Sorties A5 :", np.round(y_pred).flatten()) #J'utilise np.round pour éviter  d'avoir des valeurs telles  que 0.00003, 0.99997, 0.99998, 0.00001 renvoyés par la sigmoide
print("Vérités attendues      :", y_true)

xx, yy = np.meshgrid(np.linspace(0, 1, 200),
                     np.linspace(0, 1, 200))
grid = np.c_[xx.ravel(), yy.ravel()]  # Grille de tous les points [A1, A2]
Z = mlp.forward(grid).reshape(xx.shape)  # Prédictions sur la grille

# Tracer les zones : vert = 1, rouge = 0
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1],
             cmap=ListedColormap(['#FFBBBB', '#BBFFBB']),
             alpha=0.6)

# Ajouter les 4 points XOR (entrées connues)
plt.scatter(X[:, 0], X[:, 1], c=y_true,
            cmap=ListedColormap(['red', 'green']),
            edgecolors='k', s=150)

plt.xlabel("Entrée A1")
plt.ylabel("Entrée A2")
plt.title("Frontière de décision du Perceptron XOR")
plt.grid(True)
plt.show()
