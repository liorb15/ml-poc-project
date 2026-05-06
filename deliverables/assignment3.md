# Assignment 3

## Introduction
Ce troisième livrable présente la partie **modélisation** du projet.

L’objectif est simple : à partir des informations extraites automatiquement depuis des partitions de piano au format MusicXML, nous voulons prédire le **niveau de difficulté** d’un morceau.

Ce document explique de manière complète :
- quels modèles ont été testés ;
- comment ils ont été évalués ;
- quels résultats ont été obtenus ;
- quel modèle fonctionne le mieux pour l’instant ;
- pourquoi ce résultat doit malgré tout être interprété avec prudence.

L’idée n’est pas seulement d’afficher des scores, mais de montrer une vraie démarche de comparaison entre plusieurs modèles, avec une lecture critique des résultats.

---

## 1. Rappel du problème posé
Le projet cherche à répondre à la question suivante :

> peut-on estimer automatiquement la difficulté d’un morceau de piano à partir de sa partition ?

Nous ne partons pas d’enregistrements audio, mais de **partitions symboliques**. Cela signifie que le modèle ne “regarde” pas un son : il utilise des informations structurées extraites depuis la partition, comme :
- le nombre de notes ;
- la présence de silences ;
- la densité d’écriture ;
- la taille des sauts entre notes ;
- la variété rythmique ;
- la complexité de certaines signatures ou altérations.

La cible à prédire est une version regroupée du niveau de difficulté.

---

## 2. Rappel rapide du dataset utilisé
Le dataset actuellement utilisé est **Mikrokosmos-difficulty**.

Après nettoyage :
- 153 lignes au départ dans les métadonnées ;
- 6 pièces à quatre mains retirées ;
- 147 morceaux solo conservés.

La variable cible finale est regroupée en 3 classes :
- `beginner`
- `intermediate`
- `advanced`

Répartition finale :
- `beginner` : 92 morceaux
- `intermediate` : 45 morceaux
- `advanced` : 10 morceaux

Cette répartition est importante pour comprendre les résultats : la classe `advanced` est petite, donc plus difficile à prédire correctement.

---

## 3. Représentation des données pour les modèles
Chaque morceau est transformé en une ligne de tableau.

Autrement dit :
- **une ligne** = un morceau ;
- **les colonnes** = des caractéristiques numériques extraites de la partition ;
- **la cible** = le niveau de difficulté.

Dans `data.py`, la séparation entre les données d’entrée et la cible est faite explicitement :
- `X` = la matrice de features ;
- `y` = la colonne `difficulty_label`.

Le pipeline garde actuellement **24 features principales** dans l’entraînement.

Exemples de variables utilisées :
- `notes_total`
- `notes_played`
- `measures`
- `chord_notes`
- `pitch_span`
- `rest_ratio`
- `rhythmic_variety`
- `notes_per_pitch_class`
- `duration_cv_proxy`
- `accidental_ratio`
- `key_signature_complexity`
- `book_code`

Le but est de donner au modèle plusieurs indices indirects de difficulté, sans prétendre qu’une seule variable suffit à la résumer.

---

## 4. Séparation train/test et protocole d’évaluation
Pour évaluer les modèles proprement, nous avons utilisé deux niveaux d’évaluation.

### 4.1 Split train/test
Le dataset est d’abord séparé en deux parties :
- un ensemble d’entraînement ;
- un ensemble de test.

Le split est réalisé avec :
- `test_size = 0.2`
- `random_state = 42`
- `stratify = y`

Le fait d’utiliser `stratify=y` permet de conserver une répartition des classes cohérente entre entraînement et test.

### 4.2 Validation croisée
En plus du test final, nous avons aussi utilisé une **validation croisée stratifiée en 5 folds** sur la partie entraînement.

Concrètement, cela permet de ne pas dépendre d’un seul découpage des données. Au lieu de regarder un seul score, on mesure aussi une performance moyenne sur plusieurs sous-splits du jeu d’entraînement.

C’est particulièrement utile ici, car le dataset reste petit.

---

## 5. Métriques retenues
Deux métriques principales sont utilisées.

### 5.1 Accuracy
L’accuracy mesure la proportion totale de prédictions correctes.

C’est une métrique utile, mais elle peut être trompeuse si les classes sont déséquilibrées.

### 5.2 Macro F1
La macro F1 calcule la qualité moyenne de prédiction en donnant le même poids à chaque classe.

Cette métrique est particulièrement importante ici, car :
- la classe `advanced` est rare ;
- on ne veut pas qu’un bon score global masque un mauvais comportement sur les morceaux difficiles.

Dans ce projet, la **macro F1** est donc la métrique la plus informative pour comparer les modèles.

---

## 6. Les trois modèles comparés
Pour respecter l’objectif de comparaison entre plusieurs approches, trois modèles ont été entraînés.

### 6.1 Logistic Regression
La régression logistique sert ici de **baseline simple et lisible**.

Elle est utile parce que :
- elle donne un point de départ clair ;
- elle est assez simple à interpréter ;
- elle fonctionne souvent correctement sur de petits jeux de données tabulaires.

Dans notre pipeline, elle est utilisée avec :
- un `StandardScaler` ;
- un équilibrage des classes (`class_weight='balanced'`).

### 6.2 SVM (RBF)
Le SVM avec noyau RBF est un modèle plus souple.

L’idée derrière ce choix est de tester une méthode capable de mieux séparer les classes lorsque la frontière entre elles n’est pas linéaire.

Comme pour la régression logistique, les données sont standardisées avant apprentissage.

### 6.3 Random Forest
La Random Forest est un ensemble d’arbres de décision.

Ce modèle est intéressant ici parce qu’il peut mieux capturer :
- des interactions entre variables ;
- des relations non linéaires ;
- des combinaisons de signaux plus complexes qu’un modèle linéaire simple.

Dans ce projet, c’est actuellement le modèle qui se comporte le mieux parmi les trois baselines.

---

## 7. Résultats obtenus
Les résultats enregistrés sont les suivants.

### 7.1 Logistic Regression
- holdout accuracy : **0.8667**
- holdout macro F1 : **0.5910**
- cross-validation accuracy mean : **0.7866**
- cross-validation accuracy std : **0.0367**
- cross-validation macro F1 mean : **0.6576**
- cross-validation macro F1 std : **0.1186**

### 7.2 SVM (RBF)
- holdout accuracy : **0.7333**
- holdout macro F1 : **0.5051**
- cross-validation accuracy mean : **0.7783**
- cross-validation accuracy std : **0.0980**
- cross-validation macro F1 mean : **0.5601**
- cross-validation macro F1 std : **0.0957**

### 7.3 Random Forest
- holdout accuracy : **0.8667**
- holdout macro F1 : **0.7481**
- cross-validation accuracy mean : **0.8033**
- cross-validation accuracy std : **0.0931**
- cross-validation macro F1 mean : **0.6855**
- cross-validation macro F1 std : **0.1465**

---

## 8. Lecture simple des résultats
Si on lit ces résultats sans jargon inutile, on peut retenir trois choses.

### 8.1 Le SVM est le moins convaincant ici
Il remplit son rôle de troisième modèle comparé, mais ses scores sont plus faibles que ceux des deux autres.

Cela ne veut pas dire qu’un SVM est un mauvais modèle en général. Cela veut simplement dire que, **sur ce dataset précis**, il est moins adapté que les alternatives testées.

### 8.2 La Logistic Regression est une baseline correcte, mais limitée
Elle obtient un bon score d’accuracy sur le holdout, mais une macro F1 plus faible que la Random Forest.

Autrement dit, elle peut sembler correcte globalement, mais elle gère moins bien l’équilibre entre les trois classes.

### 8.3 La Random Forest est la meilleure baseline actuelle
C’est le modèle qui obtient la meilleure macro F1 sur le test final, et aussi la meilleure moyenne de macro F1 en validation croisée parmi les trois modèles comparés.

C’est donc le meilleur choix pratique à ce stade.

---

## 9. Pourquoi la Random Forest semble mieux marcher
Le projet repose sur des variables tabulaires construites à la main à partir des partitions.

Dans ce contexte, la Random Forest semble mieux fonctionner parce qu’elle peut exploiter plus facilement :
- des relations non linéaires ;
- des effets combinés entre plusieurs features ;
- des différences de comportement entre sous-groupes de morceaux.

Cela correspond bien à notre cas, car la difficulté d’un morceau ne dépend pas d’un seul facteur, mais d’un mélange de densité, rythme, registre, accords, variété, etc.

---

## 10. Analyse qualitative des erreurs
L’analyse actuelle montre une tendance assez claire :
- les morceaux `beginner` sont plutôt bien reconnus ;
- les morceaux `intermediate` sont globalement mieux gérés que prévu ;
- les morceaux `advanced` restent les plus difficiles à classer correctement.

Ce résultat est logique, car la classe `advanced` contient seulement 10 morceaux.

Le modèle a donc moins d’exemples pour apprendre ce qui caractérise vraiment un morceau difficile.

Cela veut dire que la limite principale n’est pas seulement “le modèle pourrait être meilleur”, mais aussi “les données pour certaines classes sont trop peu nombreuses”.

---

## 11. Importance relative de certaines features
La Random Forest ne donne pas juste un score : elle permet aussi d’observer quelles variables semblent porter le plus de signal.

Parmi les variables qui ressortent le plus dans l’analyse actuelle, on retrouve notamment :
- `measures`
- `notes_total`
- `notes_played`
- `notes_per_pitch_class`
- `accidental_ratio`
- `rhythmic_variety`
- `book_code`
- `duration_cv_proxy`

Cela renforce une idée importante :
le modèle ne repose pas sur une seule dimension de la partition, mais sur un ensemble de signaux complémentaires.

---

## 12. Essai d’amélioration avec Optuna
Une fois la meilleure baseline identifiée, nous avons ajouté une étape d’optimisation automatique des hyperparamètres avec **Optuna** sur la Random Forest.

### 12.1 À quoi sert Optuna ici ?
Optuna permet de tester automatiquement plusieurs configurations d’un modèle pour chercher celles qui donnent les meilleurs résultats.

Au lieu de choisir les réglages à la main, on laisse un outil explorer plusieurs combinaisons de paramètres.

### 12.2 Ce qui a été fait
Un script séparé a été ajouté pour cette étape.

Le tuning a été lancé sur la Random Forest avec pour objectif de maximiser la **macro F1 en validation croisée**.

Après **20 essais**, la meilleure configuration trouvée est :
- `n_estimators = 400`
- `max_depth = 10`
- `min_samples_leaf = 4`
- `max_features = log2`

La meilleure macro F1 moyenne trouvée dans ce tuning est d’environ :
- **0.7253**

### 12.3 Comment interpréter ce résultat
Ce résultat est intéressant, car il montre qu’on peut probablement améliorer la configuration de la Random Forest au-delà des choix manuels de départ.

Mais il faut rester rigoureux :
- ce tuning a été lancé dans un protocole séparé ;
- il ne faut donc pas comparer ce score trop brutalement au tableau baseline comme si tout venait exactement du même protocole d’évaluation.

Autrement dit :
Optuna montre une **piste d’amélioration crédible**, mais cette étape doit être lue comme un prolongement du travail, pas comme une preuve définitive que “le problème est résolu”.

---

## 13. Sauvegarde des modèles : joblib et compatibilité pickle
Le professeur a mentionné `joblib` et `pickle`, et cette partie a bien été prise en compte dans le projet.

### 13.1 Ce qui est utilisé en pratique
Les modèles entraînés sont actuellement **sauvegardés avec `joblib`**.

Exemples de fichiers produits :
- `mikrokosmos_log_reg.joblib`
- `mikrokosmos_svm_rbf.joblib`
- `mikrokosmos_random_forest.joblib`

### 13.2 Ce qui est supporté dans le code
Le module de chargement du projet supporte :
- `.joblib`
- `.pkl`
- `.pickle`

Donc :
- en pratique, nous utilisons **joblib** pour sauvegarder les modèles ;
- mais le code reste compatible avec des fichiers **pickle** si besoin.

Cette partie répond à une exigence utile du projet : entraîner des modèles, puis pouvoir les recharger proprement sans devoir les recalculer à chaque fois.

---

## 14. Limites actuelles de cette comparaison
Même si la comparaison de modèles est sérieuse, elle doit être lue avec recul.

### 14.1 Dataset petit
147 morceaux, c’est suffisant pour un prototype, mais encore limité pour des conclusions fortes.

### 14.2 Classe `advanced` trop petite
Avec seulement 10 morceaux, cette classe reste fragile à modéliser correctement.

### 14.3 Corpus peu diversifié
Tous les morceaux viennent de *Mikrokosmos* de Bartók.

Le modèle apprend donc sur un univers pédagogique et stylistique assez homogène.

### 14.4 Résultats encore liés au corpus utilisé
Certaines features, comme `book_code`, capturent aussi la structure interne du corpus lui-même. Cela peut aider les performances, mais cela veut aussi dire que le modèle profite partiellement de l’organisation particulière du dataset.

---

## 15. Ce qu’on peut conclure honnêtement
À ce stade, on peut dire de manière raisonnable que :
- le pipeline de modélisation fonctionne ;
- les trois modèles ont bien été comparés ;
- la **Random Forest** est la meilleure baseline actuelle ;
- l’optimisation avec **Optuna** montre une piste sérieuse d’amélioration ;
- la principale limite du projet vient désormais davantage des **données** que du simple choix du modèle.

Autrement dit, le projet ne manque pas seulement de “meilleurs algorithmes”. Il manque surtout d’un corpus plus riche, plus grand et plus varié pour aller plus loin.

---

## Conclusion
Ce troisième livrable montre une vraie étape de modélisation complète :
- définition d’un problème clair ;
- séparation explicite entre les features `X` et la cible `y` ;
- entraînement de **trois modèles différents** ;
- évaluation par **test final** et **validation croisée** ;
- comparaison critique des résultats ;
- première optimisation du meilleur modèle avec **Optuna** ;
- prise en compte de la sauvegarde des modèles avec **joblib**.

Le meilleur modèle actuel est la **Random Forest**, qui offre le meilleur compromis sur ce dataset entre performance globale et qualité de prédiction équilibrée entre les classes.

Cependant, la conclusion la plus importante n’est pas seulement “la Random Forest gagne”.
La conclusion la plus utile est plutôt la suivante :

> le pipeline est déjà solide, mais la prochaine vraie amélioration viendra probablement plus d’un meilleur dataset que d’une simple multiplication des réglages ou des features.
