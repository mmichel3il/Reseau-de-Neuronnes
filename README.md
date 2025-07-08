# R-seau-de-Neuronnes
### Exercice 1 : 
- Heaviside : une fonction échelon qui renvoie 0 pour z<0 sinon 1  
Dérivée nulle partout  
<img width="500" alt="Capture d’écran 2025-07-08 à 22 14 04" src="https://github.com/user-attachments/assets/72950960-34f2-42a3-b66e-1fdc807a59e5" />

- Sigmoid : Fonction logistique, bornée entre 0 et 1  
Dérivée σ′(z)=σ(z)(1−σ(z))  
<img width="500" alt="Capture d’écran 2025-07-08 à 22 14 47" src="https://github.com/user-attachments/assets/96af1f0f-1dc2-4a2c-9b91-38329153a2d5" />

- Tanh : variante centrée de la sigmoide, bornée entre -1 et 1  
Dérivée tanh′(z)=1−tanh^2(z)  
<img width="500" alt="Capture d’écran 2025-07-08 à 22 16 58" src="https://github.com/user-attachments/assets/9b9dcfbb-0294-4b12-b9f2-c54d2d0d892e" />

- Relu : Unité linéaire rectifiée, renvoie max(0,z)  
Dérivée nulle pour z<0 sinon 1  
<img width="500" alt="Capture d’écran 2025-07-08 à 22 17 43" src="https://github.com/user-attachments/assets/22d5b78e-aa4d-4d49-8a1b-eda605524284" />

- Leaky Relu : variante de Relu qui autorise un petit flux pour z<0
Dérivée constante pour z<0 sinon 1  
<img width="521" alt="Capture d’écran 2025-07-08 à 22 18 18" src="https://github.com/user-attachments/assets/8df2d6f0-c009-4bea-a5fd-6699331b71ef" />

On utilise numpy pour vectoriser les calculs et matplotlib pour visualiser les résultats.  
La différence entre : 
- np.array on déclare les valeurs de z dans le tableau 
- np.linspace il représente un ensemble de points sur lesquels on veut évaluer la fonction.

### Exercice 2 : Questions d'analyse :
1 – Pourquoi la fonction de Heaviside pose-t-elle problème pour l'apprentissage par gradient ?  
Elle n’est pas dérivable, donc on ne peut pas calculer les gradients.  
2 – Dans quels cas utiliser sigmoid vs tanh ?
Sigmoid pour les sorties entre 0 et 1.  
Tanh pour les couches cachées, car elle est centrée sur zéro.  
3 – Pourquoi ReLU est-elle si populaire dans les réseaux profonds ?  
Elle est rapide, ne sature pas, et facilite l’apprentissage.  
4 – Quel est l’avantage du Leaky ReLU ?  
Elle évite que les neurones restent bloqués à zéro.  

### Exercice 3 : Questions d'analyse :
1- Que se passe-t-il si le η est trop grand ?  
Les poids sont mis à jour trop brutalement ce qui fait qu' il peut sauter la bonne solution et au lieu de diminuer progressivement, l’erreur peut devenir de plus en plus grande.  
2- Et s’il est trop petit ?  
Le Perceptron dans ce cas va donc prendre plus de temps à apprendre parce que les mises à jour des poids sont minuscules ça va donc ralentir le processus et même risquer de le bloquer à un minimum local.  
3- Existe-t-il une valeur idéale de η ?  
Il n’existe pas de valeur idéale de η, elle va dépendre des données que nous avons et du modèle choisi.  
4- Peut-on faire varier η au cours du temps ?  
On peut faire varier η au cours du temps au travers de la stratégie du learning rate scheduling.  
5- Quelle stratégie pouvez vous imaginer ?  
Cette stratégie consiste à commencer avec un learning rate élevé pour apprendre vite puis le réduire progressivement pour affiner le η et pas sauter la solution.  

### Exercice 5 : Pour chaque cas :
1-Combien d’époques sont nécessaires pour converger ?  
Le nombre d’époques nécessaires ne peut pas être défini car il n’est jamais identique.  
2-Visualisez la droite de séparation trouvée  
3-Le perceptron converge-t-il toujours vers la même solution ?  
Non, pas du tout, il ne converge pas toujours vers la même solution.  

### Exercice 7 :  
La droite tracée correspond à la frontière où le perceptron décide du changement de classe. En lançant plusieurs fois le programme, elle reste identique car les données sont toujours les mêmes. Cela vient du fait que la graine aléatoire (np.random.seed(42)) fige le tirage des points. Pour voir des variations, il faudrait la retirer ou changer sa valeur afin de générer des données différentes à chaque exécution.  

### Exercice 8 : Les constatations devront renforcer, confirmer ou invalider vos réponses de l'exercice 3.
1- Lorsque n est très petit, il prends plus de temps à converger.  
2- Lorsque n est trop grand il converge plus vite mais est moins précis.  
3- Il n'existe pas de learning rate idéal mais en faisant plusieurs test on pourrait trouver une valeur du learning rate optimale.  
4- Lorsque le bruit est faible les points sont centrés sur leurs classes donc plus facile pour le perceptron separer les classes lorsque il est grand les points sont séparés donc suceptibles d'etre mélangés donc plus difficile pour le perceptron de les séparer et de construire la droite.  

### Exercice 9 : Questions d'analyse :
1-Le modèle attribue l’exemple à la classe ayant le score le plus élevé, même si plusieurs perceptrons sont positifs. C’est la compétition entre scores qui départage.  
2-Même si aucun perceptron ne “active” franchement, le modèle choisit la classe au score le plus élevé, ce qui permet d'éviter les cas “sans réponse”.  
3-L’approche “Un contre Tous” introduit un déséquilibre car la classe positive est minoritaire. Ce biais peut impacter l’apprentissage, surtout si les données sont mal réparties. Pour y remédier, on pourrait utiliser un équilibrage des classes ou adapter le taux d’apprentissage.  

### Questions de réflexion et d'analyse :
1 – Convergence : Dans quelles conditions le perceptron est-il garanti de converger ?  
Le perceptron converge seulement si les données sont linéairement séparables. Sinon, il peut tourner indéfiniment sans trouver de solution. 
2 – Initialisation : L'initialisation des poids influence-t-elle la solution finale ?  
Oui, elle influence le temps de convergence, mais pas la solution finale si les données sont bien séparables. Avec des données non séparables, l’init ne suffit pas.  
3 – Taux d'apprentissage : Comment choisir le taux d'apprentissage optimal ?
Il faut le tester. Trop petit : lent à apprendre. Trop grand : instable. Un compromis comme 0.1 est souvent un bon départ.  
4 – Généralisation : Comment évaluer la capacité de généralisation du perceptron ?  
On teste le modèle sur des données nouvelles (non vues) et on observe sa précision. Un bon score hors entraînement indique une bonne généralisation.  
5 – XOR Revisité : Proposez des solutions pour résoudre le problème XOR  
Le perceptron simple ne peut pas résoudre XOR. Il faut ajouter au moins une couche cachée pour capturer la non-linéarité du problème.  
6 – Données bruitées : Comment le perceptron se comporte-t-il avec des données bruitées ?  
Il devient instable. Les erreurs se répètent et l’algorithme peut ne jamais converger. Il n’est pas robuste au bruit.  
7 – Classes déséquilibrées : Que se passe-t-il si une classe est très minoritaire ?  
Le perceptron peut favoriser la classe majoritaire, car les mises à jour sont dominées par elle. Résultat : mauvaise performance sur la classe minoritaire.  
8 – Normalisation : Faut-il normaliser les données avant l'entraînement ?  
Oui, c’est recommandé. Des valeurs trop grandes peuvent ralentir l’apprentissage ou déséquilibrer la mise à jour des poids.  
