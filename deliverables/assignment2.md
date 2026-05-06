# Assignment 2

## Introduction
Ce deuxième livrable détaille la construction du dataset final utilisé pour le projet de prédiction de difficulté pianistique, ainsi que l’ensemble du travail de **feature engineering** réalisé à partir des partitions symboliques.

L’objectif n’est pas seulement de dire quelles colonnes sont utilisées, mais d’expliquer :
- de quel dataset brut on est parti ;
- quelles étapes de nettoyage et de filtrage ont été appliquées ;
- comment les partitions ont été transformées en variables exploitables par un modèle ;
- pourquoi certaines features ont été retenues dans le pipeline final, alors que d’autres sont restées exploratoires.

---

## 1. Dataset source
Le projet part du dataset **Mikrokosmos-difficulty**, utilisé comme base de travail principale pour cette première version du prototype.

Ce choix a été fait parce que :
- le dataset est directement accessible ;
- il contient des partitions de piano sous format **MusicXML** ;
- il fournit des informations de difficulté via le champ `henle_difficulty` ;
- il permet de construire rapidement un pipeline complet de bout en bout, du parsing des partitions jusqu’à l’entraînement des modèles.

### Pourquoi les fichiers MusicXML sont essentiels
Les fichiers **MusicXML** servent de représentation symbolique de la partition. Concrètement, ce sont eux qui permettent d’extraire automatiquement des informations musicales structurées, par exemple :
- le nombre de notes ;
- la présence de silences ;
- la durée des notes ;
- les accords ;
- les hauteurs jouées ;
- l’étendue du registre ;
- les signatures rythmiques ;
- les indications de tempo ;
- certains éléments de tonalité.

Autrement dit, **le CSV de métadonnées ne suffit pas à lui seul** pour entraîner le modèle. Il contient les labels et quelques informations descriptives, mais ce sont les fichiers MusicXML qui rendent possible le **feature engineering musical**. Sans eux, on ne pourrait pas reconstruire la structure interne de chaque morceau ni produire les variables expliquant sa difficulté technique.

---

## 2. Nettoyage et constitution du dataset de travail
Le dataset brut ne peut pas être injecté tel quel dans le pipeline.

### 2.1 Taille du dataset brut
Le fichier de métadonnées contient initialement **153 lignes**.

### 2.2 Filtrage des pièces non retenues
Toutes les entrées n’ont pas été gardées. Les pièces à **quatre mains** n’ont pas été retenues, car elles ne correspondent pas au cadre visé ici : la prédiction de difficulté d’un morceau de **piano solo**.

Bilan après filtrage :
- total initial : **153** lignes ;
- pièces à quatre mains exclues : **6** ;
- pièces solo restantes : **147**.

### 2.3 Vérification des partitions exploitables
Pour chaque ligne conservée, il faut également vérifier que le fichier MusicXML correspondant existe bien dans le dossier `musicxml/`.

Dans notre cas, les **147 pièces solo conservées** possèdent bien un fichier XML exploitable. Le dataset de travail final contient donc **147 morceaux**.

---

## 3. Construction de la variable cible
Le dataset source fournit une colonne `henle_difficulty` avec des valeurs textuelles telles que :
- `Piano 1 easy`
- `Piano 2 easy`
- `Piano 3 easy`
- `Piano 4 medium`
- `Piano 5 medium`
- `Piano 6 medium`
- etc.

### 3.1 Problème posé par le label brut
Le label initial est utile, mais trop fin pour ce prototype :
- le dataset est petit ;
- certaines classes supérieures sont peu représentées ;
- un nombre trop élevé de niveaux rendrait l’apprentissage plus instable.

### 3.2 Regroupement en classes plus robustes
Le label a donc été transformé en **trois classes de difficulté** :
- niveaux 1 à 3 → `beginner`
- niveaux 4 à 6 → `intermediate`
- niveaux 7 et plus → `advanced`

### 3.3 Répartition finale des classes
Après transformation, la distribution du dataset final est la suivante :
- `beginner` : **92** morceaux
- `intermediate` : **45** morceaux
- `advanced` : **10** morceaux

Ce regroupement rend la tâche plus stable, mais il met aussi en évidence une limite importante : la classe `advanced` reste peu représentée.

---

## 4. Représentation finale du dataset
Après parsing des partitions et extraction des variables, le dataset final devient une **table de modélisation** dans laquelle :
- **une ligne = un morceau** ;
- les colonnes contiennent des **features numériques** extraites des partitions ;
- la cible est la variable `difficulty_label`.

Le dataset final utilisé pour l’apprentissage contient :
- **147 lignes** ;
- **24 features finales** retenues dans le pipeline principal ;
- **6 features exploratoires** calculées mais non retenues dans l’entraînement principal ;
- **aucune valeur manquante** sur les features construites.

---

## 5. Méthodologie générale de feature engineering
L’idée générale du feature engineering est la suivante : la difficulté pianistique n’est pas directement observable comme une variable simple, donc il faut la **proxifier** par plusieurs dimensions musicales calculables à partir des partitions.

Nous avons cherché à décrire plusieurs aspects de la difficulté :
- la **quantité** de matière musicale ;
- la **densité** d’écriture ;
- la **complexité rythmique** ;
- la **texture harmonique** ;
- l’**étendue et la mobilité** du registre ;
- certains indices de **complexité tonale et métrique**.

Le but n’est pas de prétendre qu’une feature résume à elle seule la difficulté, mais de combiner plusieurs indices complémentaires.

---

## 6. Features retenues dans le pipeline final
Les 24 features suivantes sont celles retenues dans `load_dataset_split()` pour l’entraînement des modèles.

### 6.1 Features de volume et de densité globale
#### `notes_total`
Nombre total de nœuds `<note>` dans la partition, y compris les silences.

**Intuition :** un morceau plus long ou plus chargé peut être plus exigeant à lire et à interpréter.

#### `notes_played`
Nombre de notes effectivement jouées, en excluant les rests.

**Intuition :** plus il y a d’événements joués, plus la charge pianistique potentielle augmente.

#### `rests`
Nombre de silences détectés dans la partition.

**Intuition :** les silences peuvent alléger ou structurer l’écriture ; leur fréquence contribue au profil rythmique global.

#### `measures`
Nombre de mesures dans le morceau.

**Intuition :** une pièce plus longue peut demander davantage de mémorisation, de contrôle formel et de continuité technique.

#### `notes_per_measure`
Nombre moyen de notes jouées par mesure.

**Intuition :** cette variable capture une densité plus informative que le seul nombre total de notes.

---

### 6.2 Features rythmiques
#### `avg_duration`
Durée moyenne des notes.

**Intuition :** une écriture dominée par des durées courtes peut indiquer plus de mobilité et de précision d’exécution.

#### `duration_std`
Écart-type des durées de notes.

**Intuition :** plus les durées varient, plus le contrôle rythmique peut être exigeant.

#### `duration_cv_proxy`
Version normalisée de la variabilité des durées, calculée à partir de `duration_std` et `avg_duration`.

**Intuition :** cela permet d’évaluer la variabilité rythmique en tenant compte du niveau global de durée moyenne, plutôt que de comparer uniquement des écarts bruts.

#### `rhythmic_variety`
Nombre de durées distinctes présentes dans la partition.

**Intuition :** plus le vocabulaire rythmique est varié, plus la lecture et l’exécution peuvent être complexes.

#### `rest_ratio`
Proportion de silences parmi les événements de type note.

**Intuition :** ce ratio affine l’analyse du rythme en tenant compte de l’alternance jeu / silence plutôt qu’en s’arrêtant à des comptes bruts.

---

### 6.3 Features de texture harmonique et d’accords
#### `chord_notes`
Nombre de notes faisant partie d’un accord, détectées via la balise `<chord>`.

**Intuition :** la présence d’accords augmente souvent la difficulté de coordination, de précision et d’équilibre sonore.

#### `chord_ratio`
Proportion de notes jouées qui appartiennent à un contexte d’accord.

**Intuition :** ce ratio est plus informatif que le compte brut, car il tient compte de la taille globale du morceau.

---

### 6.4 Features de registre, de variété de hauteurs et de mobilité mélodique
#### `pitch_span`
Différence entre la note la plus grave et la note la plus aiguë du morceau.

**Intuition :** une grande étendue de clavier peut traduire une difficulté technique plus importante, notamment en termes de déplacement et de contrôle spatial.

#### `unique_pitch_count`
Nombre de hauteurs distinctes utilisées.

**Intuition :** une plus grande variété de notes peut correspondre à une écriture plus riche et plus difficile à stabiliser.

#### `avg_pitch_interval`
Intervalle moyen entre notes successives jouées.

**Intuition :** des déplacements moyens plus grands peuvent traduire une mobilité mélodique plus exigeante.

#### `max_pitch_interval`
Plus grand saut observé entre deux notes successives.

**Intuition :** les grands sauts sont souvent associés à une difficulté technique plus élevée.

---

### 6.5 Features normalisées et densités dérivées
#### `notes_per_pitch_class`
Rapport entre le nombre de notes jouées et le nombre de hauteurs distinctes.

**Intuition :** cette feature mesure une densité relative du matériau musical. Elle permet d’aller au-delà du simple “beaucoup de notes” en tenant compte de la diversité de l’espace tonal utilisé.

#### `notes_per_measure_per_pitch_class`
Densité moyenne par mesure, normalisée par la variété de hauteurs.

**Intuition :** cette variable a été conçue pour mieux capturer la charge musicale locale tout en évitant qu’un morceau long ou très répétitif soit artificiellement favorisé.

---

### 6.6 Features de contexte symbolique
#### `tempo_mean`
Tempo moyen extrait des balises `.//sound[@tempo]` quand elles sont disponibles.

**Intuition :** un tempo plus élevé peut augmenter la contrainte d’exécution, surtout lorsqu’il se combine à une forte densité de notes.

#### `notes_per_second_proxy`
Approximation de la densité temporelle, calculée à partir du nombre de notes jouées et du tempo moyen.

**Intuition :** cette variable cherche à représenter la quantité de matière à gérer dans une fenêtre temporelle donnée.

#### `accidental_ratio`
Proportion de notes impliquant des altérations hors classe diatonique de base.

**Intuition :** une écriture avec davantage d’altérations peut refléter une complexité tonale ou digitale plus élevée.

#### `key_signature_complexity`
Complexité moyenne de tonalité, approchée à partir de la valeur absolue de `fifths` dans les signatures d’armure.

**Intuition :** certaines armures plus éloignées de Do majeur / La mineur peuvent être plus coûteuses à lire ou à anticiper.

#### `time_signature_changes`
Nombre de changements de signature rythmique repérés dans la pièce.

**Intuition :** les changements métriques peuvent compliquer la lecture, la pulsation et la structuration du morceau.

#### `book_code`
Encodage du volume de *Mikrokosmos* auquel appartient le morceau :
- Volumes I-II → 1
- Volumes III-IV → 2
- Volumes V-VI → 3

**Intuition :** dans ce corpus précis, la progression entre volumes reflète déjà en partie une montée de difficulté. Cette feature agit comme un signal contextuel structurant propre au dataset.

---

## 7. Features exploratoires calculées mais non retenues dans le pipeline principal
Certaines features ont été calculées dans le dataframe brut, mais n’ont pas été gardées dans la liste finale de `feature_columns`.

### `pitch_std`
Écart-type des hauteurs jouées.

**Idée :** mesurer la dispersion du registre plus finement que `pitch_span`.

**Pourquoi non retenue :** plausible musicalement, mais pas assez convaincante dans les expérimentations par rapport à d’autres variables plus simples.

### `large_leap_ratio`
Proportion des intervalles successifs supérieurs ou égaux à un grand saut.

**Idée :** isoler les déplacements les plus difficiles.

**Pourquoi non retenue :** l’intuition est bonne, mais la variable n’a pas suffisamment amélioré les résultats du pipeline principal.

### `avg_chord_cluster_size`
Taille moyenne des grappes d’accords détectées.

**Idée :** aller plus loin que le simple nombre de notes en accord.

**Pourquoi non retenue :** intéressante sur le plan musical, mais non suffisamment robuste sur ce petit corpus.

### `span_per_unique_pitch`
Étendue du registre normalisée par la variété de hauteurs.

**Idée :** distinguer les pièces très étendues mais peu variées de celles qui le sont davantage.

**Pourquoi non retenue :** utile pour l’exploration, mais pas encore justifiée dans la version finale.

### `notes_per_second_per_pitch_class`
Densité temporelle normalisée par la variété des hauteurs.

**Idée :** raffiner la mesure de difficulté en combinant tempo, quantité et diversité.

**Pourquoi non retenue :** variation réelle dans le dataset, mais gain insuffisant dans la configuration finale.

### `tempo_duration_interaction`
Interaction entre tempo et variabilité rythmique.

**Idée :** mesurer le fait qu’une écriture rythmique instable devienne encore plus difficile à tempo élevé.

**Pourquoi non retenue :** hypothèse intéressante mais trop fragile à ce stade sur un dataset de taille limitée.

---

## 8. Sélection des features : raisonnement adopté
Le travail de sélection n’a pas consisté à empiler le plus grand nombre de variables possible. Au contraire, l’objectif a été de distinguer :
- les features qui apportent une information plausible et utile ;
- les features redondantes ;
- les features séduisantes sur le plan intuitif mais peu robustes empiriquement.

L’enseignement principal de cette phase est le suivant : **plus de features n’implique pas automatiquement un meilleur modèle**. Certaines variables simples, bien normalisées et cohérentes avec la structure du corpus, se sont révélées plus utiles que des constructions plus sophistiquées.

Parmi les variables semblant porter le plus de signal dans le meilleur modèle actuel, on retrouve notamment :
- `measures`
- `notes_total`
- `notes_played`
- `notes_per_pitch_class`
- `accidental_ratio`
- `rhythmic_variety`
- `book_code`
- `duration_cv_proxy`

---

## 9. Expérience complémentaire : test d’une PCA
Une réduction de dimension par **PCA** a été testée pour vérifier si un espace latent plus compact permettrait une meilleure généralisation.

### Résultat observé
- certaines configurations, notamment pour la régression logistique, ont amélioré la **macro-F1 moyenne en validation croisée** ;
- en revanche, ces mêmes configurations ont **dégradé les performances sur le holdout set** ;
- la meilleure performance pratique reste obtenue par la **Random Forest sans PCA**.

### Interprétation
Sur ce dataset :
- la PCA peut compresser de l’information utile ;
- les features symboliques actuelles sont déjà relativement interprétables ;
- la réduction de dimension ne constitue donc pas ici une amélioration suffisamment convaincante pour être adoptée dans le pipeline principal.

La PCA est donc conservée comme **expérience documentée**, mais **non retenue** dans la version finale du modèle.

---

## 10. Performances des modèles sur le dataset final
Les métriques actuellement enregistrées distinguent désormais :
- les performances **holdout** sur le split de test final ;
- les performances **cross-validation** (5-fold stratifiée) sur le split d'entraînement.

Trois modèles de base sont actuellement comparés :
- **Logistic Regression**
- **SVM (RBF)**
- **Random Forest**

### Logistic Regression
- holdout accuracy : **0.8667**
- holdout macro F1 : **0.5910**
- cross-validation accuracy mean : **0.7866**
- cross-validation accuracy std : **0.0367**
- cross-validation macro F1 mean : **0.6576**
- cross-validation macro F1 std : **0.1186**

### SVM (RBF)
- holdout accuracy : **0.7333**
- holdout macro F1 : **0.5051**
- cross-validation accuracy mean : **0.7783**
- cross-validation accuracy std : **0.0980**
- cross-validation macro F1 mean : **0.5601**
- cross-validation macro F1 std : **0.0957**

### Random Forest
- holdout accuracy : **0.8667**
- holdout macro F1 : **0.7481**
- cross-validation accuracy mean : **0.8033**
- cross-validation accuracy std : **0.0931**
- cross-validation macro F1 mean : **0.6855**
- cross-validation macro F1 std : **0.1465**

À ce stade, la **Random Forest** reste le meilleur modèle pratique sur le dataset final retenu :
- elle garde la meilleure macro-F1 sur le holdout ;
- elle reste aussi légèrement devant en moyenne sur la macro-F1 de validation croisée ;
- le **SVM (RBF)** remplit le rôle de troisième baseline comparée, mais reste moins convaincant sur ce corpus.

### 10.1 Optimisation complémentaire avec Optuna
Pour aller au-delà des baselines manuelles, une recherche d’hyperparamètres **Optuna** a été ajoutée pour la **Random Forest**.

Ce tuning est réalisé **dans un script séparé**, en maximisant la **macro-F1 en validation croisée** sur le dataset `X, y` complet. Il s’agit donc d’une **expérience complémentaire de recherche d’hyperparamètres**, et non encore du modèle baseline principal utilisé dans le tableau précédent.

Après **20 essais**, la meilleure configuration trouvée est :
- `n_estimators = 400`
- `max_depth = 10`
- `min_samples_leaf = 4`
- `max_features = log2`

La meilleure **macro-F1 moyenne en cross-validation** obtenue par cette recherche est d’environ **0.7253**.

Ce résultat indique qu’une optimisation automatisée peut proposer une meilleure configuration candidate pour la Random Forest. En revanche, il faut rester prudent dans l’interprétation : ce score provient d’un protocole de tuning séparé et ne doit pas être comparé trop directement, chiffre contre chiffre, avec les métriques holdout / CV du tableau baseline précédent.

---

## 11. Discussion sur la qualité du dataset final
### Ce que le dataset permet
Le dataset final est suffisamment propre et structuré pour :
- tester un pipeline de parsing MusicXML ;
- construire un jeu de features symboliques crédible ;
- entraîner et comparer plusieurs modèles ;
- analyser l’apport de différentes variables musicales.

### Ce que le dataset ne permet pas encore pleinement
Le dataset reste limité sur plusieurs aspects :
- **taille réduite** : 147 morceaux seulement ;
- **déséquilibre des classes** : la classe `advanced` ne contient que 10 pièces ;
- **faible diversité stylistique** : tous les morceaux viennent de Béla Bartók ;
- **fort ancrage pédagogique** : le corpus suit une progression propre à *Mikrokosmos*.

Cela signifie que le dataset est **suffisant pour un prototype**, mais pas encore pour soutenir des conclusions de généralisation fortes à l’ensemble du répertoire pianistique.

---

## Conclusion
Le dataset final utilisé dans ce projet n’est pas simplement le dataset téléchargé au départ. Il résulte d’une chaîne de transformation complète :
- sélection des pièces pertinentes ;
- vérification des partitions exploitables ;
- transformation de la variable cible ;
- extraction de features depuis les fichiers MusicXML ;
- sélection d’un sous-ensemble de variables jugées les plus robustes pour l’apprentissage.

Le rôle des fichiers **MusicXML** est central : ce sont eux qui rendent possible l’extraction d’informations musicales fines et donc la construction du dataset final de modélisation.

Au final, cette phase de feature engineering montre que le projet repose sur une vraie transformation des données symboliques, et non sur l’utilisation brute d’un simple tableau de métadonnées. Le dataset obtenu est suffisamment riche pour soutenir un prototype sérieux de prédiction de difficulté, tout en restant limité par la taille et l’homogénéité du corpus.