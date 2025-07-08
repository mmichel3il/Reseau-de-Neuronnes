## Introduction : Du perceptron simple au réseau multicouches
Le perceptron simple est un modèle de neurone capable de résoudre des problèmes linéaires, mais il échoue sur des fonctions non linéaires comme XOR. Les réseaux multicouches, en ajoutant des couches cachées et des fonctions d’activation, permettent de modéliser des relations plus complexes et de résoudre ces problèmes. Cette évolution a grandement amélioré les capacités d’apprentissage des réseaux de neurones.
## Méthodes : Architecture des réseaux multicouches
Un réseau multicouches comprend une couche d’entrée, des couches cachées et une couche de sortie. Chaque neurone applique une transformation linéaire suivie d’une fonction d’activation non linéaire, ce qui permet d’apprendre des fonctions complexes.
## Algorithme de rétropropagation
La rétropropagation calcule les gradients de la perte par rapport aux poids en combinant une propagation avant puis une propagation arrière. Ces gradients servent à mettre à jour les poids pour réduire l’erreur.
## Fonctions de coût et d’optimisation
La fonction de coût (ici l’erreur quadratique moyenne) mesure l’écart entre prédictions et valeurs réelles. La descente de gradient ajuste les poids en minimisant cette erreur selon un taux d’apprentissage fixé.
## Résultats : 
Le réseau multicouches a réussi à résoudre le problème XOR, confirmant que deux neurones dans une couche cachée suffisent pour apprendre cette fonction non linéaire.  
Des tests sur plusieurs architectures ont montré que l’augmentation du nombre de neurones facilite la convergence, mais avec un risque de surapprentissage.  
Les expérimentations sur d’autres jeux de données synthétiques et réels ont validé la capacité du réseau à généraliser.  
Enfin, les courbes d’apprentissage ont permis d’observer la diminution progressive de la perte ainsi que l’évolution de la précision pendant l’entraînement.  
## Discussion : 
Les réseaux multicouches peuvent approximer des fonctions complexes, ce qui est un grand avantage. Cependant, ils demandent plus de calculs et risquent le sur-apprentissage, surtout avec beaucoup de neurones ou couches. Pour limiter cela, on utilise des stratégies comme la régularisation (L2, dropout) et la validation croisée.
## Conclusion : 
Le réseau multicouches permet de résoudre des problèmes non linéaires comme XOR. Une architecture adaptée et une bonne initialisation sont essentielles. À l’avenir, améliorer l’optimisation et intégrer des techniques de régularisation peut rendre le modèle plus robuste.
