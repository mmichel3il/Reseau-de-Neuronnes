## Introduction : Du perceptron simple au réseau multicouche avec rétropropagation
Le perceptron simple étant limité aux problèmes linéaires, cette branche explore l'utilisation d’un réseau multicouche pour modéliser des relations non linéaires. L’objectif est d’implémenter la rétropropagation pour ajuster les poids en profondeur et permettre au réseau d’apprendre des fonctions complexes comme XOR.
## Méthodes : Apprentissage par rétropropagation
Le réseau est constitué d’une couche d’entrée, d’une ou plusieurs couches cachées, et d’une couche de sortie. Chaque neurone applique une combinaison linéaire suivie d’une fonction d’activation non linéaire (sigmoïde). L’algorithme de rétropropagation permet de propager l’erreur en arrière et d’ajuster les poids via la descente de gradient. L’erreur quadratique moyenne est utilisée comme fonction de coût.
## Résultats :
Tests sur fonctions logiques : Le réseau parvient à apprendre des fonctions comme XOR, ce que le perceptron simple ne pouvait pas faire.  
Analyse de convergence : La perte diminue progressivement au fil des itérations, montrant que le réseau converge correctement.  
Évaluation sur données réelles : Le réseau est testé avec succès sur un petit jeu de données réel (ex : Iris), montrant sa capacité de généralisation.  
## Discussion :
Limites du perceptron multicouche : Il nécessite un bon réglage des hyperparamètres et peut surapprendre si le réseau est trop complexe.
Cas d’usage appropriés : Apprentissage supervisé sur des données non linéaires simples, classification de petites bases.
## Conclusion :
Le perceptron multicouche, grâce à la rétropropagation, surmonte les limites du perceptron simple. Il apprend des fonctions non linéaires, mais demande une architecture adaptée et une régulation soigneuse pour éviter le surapprentissage.
