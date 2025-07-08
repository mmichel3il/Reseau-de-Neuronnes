# Reseau-de-Neuronnes

### Exercice 1.1 : 
1 – Que signifie concrètement le théorème d’approximation universelle ?  
Ce théorème affirme qu’un réseau de neurones avec au moins une couche cachée et une fonction d’activation non linéaire, comme la sigmoïde, peut approximativement reproduire n’importe quelle fonction continue sur un intervalle donné, à condition d’avoir suffisamment de neurones.  
2 – Ce théorème garantit-il qu’on peut toujours trouver les bons poids ?  
Non, il ne le garantit pas. Le théorème prouve simplement que les bons poids existent pour une fonction donnée, mais il ne dit pas qu’on va les trouver facilement. En pratique, l’apprentissage peut échouer si les données sont mal choisies, si le réseau est mal paramétré ou si l’algorithme ne converge pas.  
3 – Quelle est la différence entre “pouvoir approximer” et “pouvoir apprendre” ?  
Pouvoir approximer, c’est une propriété théorique : cela signifie qu’il existe un réseau capable de reproduire une fonction. Pouvoir apprendre, c’est réussir à trouver les bons paramètres (poids et biais) à partir de données réelles. On peut donc dire que l’approximation est une possibilité, tandis que l’apprentissage est une réalisation.  
4 – Pourquoi utilise-t-on souvent beaucoup plus de couches cachées en pratique ?  
On utilise plusieurs couches cachées car elles permettent au réseau de construire des représentations plus abstraites. Chaque couche extrait des caractéristiques de plus en plus complexes. Cela rend le réseau plus puissant pour traiter des tâches comme la reconnaissance d’images, le langage ou les données non linéaires.  
5 – En principe, vous avez déjà vu au lycée un autre type d’approximateur de fonctions, donner leurs noms ?  
Oui, au lycée, on rencontre souvent des approximateurs comme les polynômes, les développements limités (séries de Taylor), et parfois les séries trigonométriques (comme les séries de Fourier). Ce sont des outils mathématiques qui permettent d’approcher des fonctions en utilisant des équations explicites.  

### Exercice 1.2 : Expliquer la phrase suivante
Le théorème d’approximation universelle affirme qu’un réseau profond peut exactement retrouver les données d’entraînement.  
Cette phrase signifie que si on donne à un réseau de neurones suffisamment de neurones et de couches, alors il est capable de produire une fonction qui passe exactement par tous les points des données d’entraînement, même si la fonction réelle est complexe.  

### Questions d'analyse : 
1 – Le réseau arrive-t-il à résoudre XOR ? Avec quelle architecture minimale ?  
Oui, le réseau parvient à résoudre le problème de XOR. Ce n’est pas possible avec un perceptron simple car la fonction XOR n’est pas linéairement séparable. En revanche, avec une architecture contenant une couche cachée composée de deux neurones et une fonction d’activation non linéaire comme la sigmoïde, le réseau est capable de séparer les cas où XOR vaut 1 ou 0. C’est donc l’architecture minimale nécessaire pour ce problème.  
2 – Comment le nombre de neurones cachés influence-t-il la convergence ?  
En augmentant le nombre de neurones cachés, on offre au réseau plus de “flexibilité” pour modéliser des fonctions complexes. Cela peut accélérer la convergence dans certains cas, surtout si les données sont bruitées ou complexes.  
3 – Que se passe-t-il avec plusieurs couches cachées ?  
En ajoutant plusieurs couches cachées, le réseau devient capable d’extraire des représentations plus abstraites. Chaque couche apprend à combiner les sorties de la couche précédente, ce qui permet de modéliser des relations non évidentes dans les données.  
4 – L'initialisation des poids a-t-elle une influence ?  
Oui,  l'initialisation des poids a un impact direct sur l’apprentissage. Si les poids sont mal choisis (trop grands, trop petits ou trop similaires), le réseau peut stagner, converger lentement ou même ne pas apprendre du tout.  
