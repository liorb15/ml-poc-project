# Assignment 1

## Mon projet
Mon projet consiste à construire un système capable de **prédire le niveau de difficulté d’un morceau de piano** à partir de caractéristiques musicales extraites de partitions numériques, puis de **recommander des morceaux similaires adaptés au niveau du pianiste**.

L’idée est donc de combiner :
- une **brique principale de machine learning** centrée sur la prédiction de difficulté ;
- une **brique secondaire de recommandation** permettant de proposer des morceaux cohérents avec les goûts de l’utilisateur tout en respectant son niveau technique.

## Le business case
Lorsqu’un pianiste, débutant ou intermédiaire, cherche un nouveau morceau à travailler, il peut assez facilement trouver des œuvres qui lui plaisent musicalement, mais il est beaucoup plus difficile de savoir si elles sont réellement adaptées à son niveau.

Ce problème apparaît dans plusieurs cas concrets :
- un élève veut jouer un morceau dans le style de Chopin, Debussy ou Bartók, mais ne sait pas si la partition est trop difficile ;
- un professeur cherche à proposer un répertoire progressif et motivant à un élève ;
- un pianiste amateur souhaite découvrir de nouveaux morceaux proches de ses goûts sans perdre du temps sur des œuvres injouables à court terme.

Le projet vise donc à répondre à une question simple mais utile : **comment recommander un morceau musicalement pertinent sans dépasser le niveau technique du pianiste ?**

D’un point de vue produit, un tel système pourrait servir à :
- mieux orienter le choix des morceaux à étudier ;
- rendre l’apprentissage plus progressif et plus motivant ;
- aider à découvrir de nouveaux compositeurs ou nouvelles œuvres sans frustration ;
- fournir une aide à la décision pour des plateformes pédagogiques, des professeurs de piano ou des applications d’apprentissage musical.

## La problématique machine learning
La tâche principale du projet sera de **prédire un niveau de difficulté** à partir de données musicales symboliques.

Selon le dataset final retenu, cette tâche pourra être formulée de deux manières :
- comme un problème de **classification** si la difficulté est représentée par des classes discrètes, par exemple `débutant`, `intermédiaire`, `avancé` ou des niveaux entiers ;
- comme un problème de **régression** si la difficulté est disponible sous forme d’un score continu ou quasi-continu.

À ce stade, l’hypothèse de travail la plus réaliste est une **classification**, car plusieurs sources potentielles utilisent déjà des niveaux de difficulté catégorisés.

La partie recommandation ne sera pas le cœur du modèle ML, mais plutôt une **couche applicative** construite à partir de métadonnées musicales et de la difficulté prédite ou connue.

## Les données que je souhaite utiliser
Je souhaite travailler à partir de **données musicales symboliques** plutôt qu’audio, car elles permettent d’extraire plus directement des éléments utiles pour la difficulté pianistique :
- nombre de notes ;
- densité de notes ;
- nombre de mesures ;
- présence d’accords ;
- étendue du clavier utilisée ;
- complexité rythmique ;
- indications de structure ou d’écriture ;
- éventuellement tonalité, signatures rythmiques ou autres informations de partition.

Les sources de données actuellement identifiées sont les suivantes.

### 1. Mikrokosmos-difficulty
Comme l’accès au dataset CIPI n’est finalement pas disponible, ce dataset devient la **base principale du projet** pour cette première version.

Intérêt principal :
- partitions **MusicXML** directement exploitables ;
- niveaux de difficulté associés ;
- structure de données propre et facile à intégrer dans le pipeline ;
- dataset suffisant pour construire un prototype complet de bout en bout.

Sa principale limite est qu’il est centré sur l’univers de **Bartók / Mikrokosmos**, donc moins riche pour une recommandation variée en termes de style et de compositeurs. En revanche, il reste tout à fait pertinent pour valider le cœur du projet : **la prédiction de difficulté pianistique à partir de partitions symboliques**.

### 2. PianoLibrary
Cette source est intéressante surtout pour la **partie recommandation**.

Intérêt principal :
- classement par difficulté ;
- organisation par compositeur ;
- grand catalogue d’œuvres de piano classique.

Cette base pourrait être utile pour enrichir le catalogue final de recommandations, même si elle n’est pas forcément idéale comme dataset principal d’entraînement.

### 3. Datasets complémentaires éventuels
D’autres sources comme **MAESTRO**, **ASAP** ou d’autres jeux de données symboliques pourraient être utilisées plus tard pour enrichir certaines métadonnées ou certaines caractéristiques musicales, si cela s’avère utile par la suite.

## Variables et informations envisagées
Les variables utilisées dépendront du dataset final, mais je souhaite m’appuyer sur des caractéristiques telles que :
- longueur du morceau ;
- nombre de notes jouées ;
- densité de notes par mesure ;
- nombre de notes simultanées / présence d’accords ;
- étendue entre les notes graves et aiguës ;
- complexité apparente de l’écriture ;
- structure du morceau ;
- compositeur ;
- période ou style si disponible ;
- niveau de difficulté annoté.

Ces variables devraient permettre de relier la structure musicale du morceau à son exigence technique pour un pianiste.

## La cible du projet
La cible principale sera donc le **niveau de difficulté du morceau**.

Selon les données disponibles, cette cible pourrait prendre la forme :
- d’un niveau entier, par exemple de 1 à 9 ;
- ou d’une version regroupée en grandes catégories comme `débutant`, `intermédiaire`, `avancé`.

Pour la phase de prototypage, il est probable qu’un regroupement en quelques classes soit plus robuste, notamment en cas de déséquilibre entre les niveaux.

## Les modèles envisagés
Je souhaite comparer plusieurs approches simples et interprétables avant d’aller vers des modèles plus complexes.

Par exemple :
- **Logistic Regression** pour une baseline claire et facile à interpréter ;
- **Random Forest** pour capturer des interactions non linéaires entre les caractéristiques ;
- éventuellement **XGBoost** ou un autre modèle de boosting si la taille et la qualité du dataset le justifient.

L’objectif n’est pas de choisir le modèle le plus complexe possible, mais de comparer plusieurs approches raisonnables sur une tâche bien définie.

## Les métriques envisagées
Comme il s’agira probablement d’un problème de classification, je compte utiliser des métriques adaptées telles que :
- l’**accuracy** ;
- le **macro F1-score** ;
- éventuellement une **matrice de confusion** pour comprendre les erreurs entre niveaux proches.

Ces métriques permettront de voir si le modèle distingue correctement les morceaux faciles, intermédiaires et difficiles.

## Première idée de workflow
1. Sélectionner le dataset principal le plus exploitable.
2. Nettoyer les données et conserver uniquement les morceaux pertinents pour le projet.
3. Extraire ou calculer des caractéristiques musicales à partir des partitions symboliques.
4. Définir la cible de difficulté sous une forme exploitable par le modèle.
5. Séparer les données en ensembles d’entraînement et de test.
6. Entraîner plusieurs modèles de classification.
7. Comparer leurs performances avec des métriques adaptées.
8. Sélectionner le modèle le plus pertinent.
9. Développer une interface Streamlit simple pour présenter le projet, les résultats et une logique de recommandation.
10. Permettre à l’utilisateur de choisir un niveau et éventuellement un style, un compositeur ou un morceau de référence afin d’obtenir des suggestions adaptées.

## Ce que montrera l’application finale
L’application Streamlit devra permettre de présenter :
- le problème étudié ;
- les données utilisées ;
- les caractéristiques extraites ;
- les performances des modèles ;
- une démonstration simple de recommandation de morceaux.

Une version réaliste de l’application pourrait proposer :
- un choix de niveau ;
- un choix de style ou de compositeur ;
- une liste de morceaux recommandés compatibles avec ces préférences.

## Limites et risques identifiés
À ce stade, plusieurs limites sont déjà identifiées :
- l’accès au dataset principal peut dépendre d’une validation ou d’une autorisation ;
- certains datasets peuvent être trop spécialisés ou déséquilibrés ;
- la difficulté musicale reste en partie une notion subjective ;
- la recommandation finale dépendra de la richesse des métadonnées réellement disponibles.

Malgré cela, le sujet reste pertinent car il combine une tâche ML claire, une application concrète et une forte cohérence avec le domaine musical.

## Remarque finale
Le **cœur scientifique du projet** sera la **prédiction de difficulté**.

La **recommandation de morceaux similaires** sera une extension applicative destinée à rendre le projet plus concret, plus utile et plus intéressant du point de vue utilisateur.

Autrement dit, même si la partie recommandation reste simple dans une première version, le projet restera solide tant que la prédiction de difficulté est bien construite, bien évaluée et clairement justifiée.