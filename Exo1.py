import numpy as np
import matplotlib.pyplot as plt


class ActivationFunction:
    def __init__(self, name, alpha=0.01):
        self.name = name.lower()
        self.alpha = alpha # Pour Leaky ReLU

    def apply(self, z):
        if self.name == "heaviside":
            return np.where(z<0,0,1)
        elif self.name == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.name == "tanh":
            return (np.exp(z)-np.exp(-z)) / (np.exp(z)+ np.exp(-z))
        elif self.name == "relu":
            return np.where(z<0,0,z)
        elif self.name == "leaky_relu":
            return np.where(z<0,self.alpha*z,z)
        else:
            raise ValueError(f"Activation '{self.name}' non reconnue.")

    def derivative(self, z):
        if self.name == "heaviside":
        # La dérivée de Heaviside est la distribution de Dirac
            return np.where(z!=0,0,0)
        elif self.name == "sigmoid":
            return 1 / (1 + np.exp(-z)) * (1-(1 / (1 + np.exp(-z))))
        elif self.name == "tanh":
            return 1-(np.exp(z)-np.exp(-z)) / (np.exp(z)+ np.exp(-z))**2
        elif self.name == "relu":
            return np.where(z<0,0,1)
        elif self.name == "leaky_relu":
            return np.where(z<0,self.alpha,1)
        else:
            raise ValueError(f"Dérivée de '{self.name}' non définie.")
            
z=np.array([-5, 0, 4.3])
z=np.linspace(-10, 10, 100)
act = ActivationFunction("tanh")
der = act.derivative(z)
plt.figure(figsize=(10, 6))
plt.xlabel('z')
plt.plot(z, act.derivative(z), label='tanhd', color='orange')
plt.plot(z, act.apply(z), label='tanh', color='blue')
