# Assignment 1

## Mon projet
Prédire le niveau de difficulté d’un morceau de piano et recommander des morceaux similaires adaptés au niveau du pianiste.

## Le business case
Lorsqu’un pianiste cherche un nouveau morceau à travailler, il peut facilement trouver des œuvres qui lui plaisent sur le plan musical, mais il est plus difficile de savoir si elles correspondent réellement à son niveau technique.

L’objectif du projet est donc de construire un système capable :
- d’estimer la difficulté d’un morceau de piano à partir de ses caractéristiques musicales ;
- de recommander ensuite des morceaux proches du style recherché, tout en restant adaptés au niveau de l’utilisateur.

Ce type d’outil pourrait être utile pour :
- aider des élèves à choisir des morceaux accessibles ;
- aider des professeurs à proposer un répertoire adapté ;
- faciliter la découverte de nouveaux morceaux sans tomber sur une œuvre trop simple ou trop difficile.

## Les sources de données
Je souhaite travailler à partir de datasets de musique symbolique et de métadonnées sur des morceaux de piano classique.

Pistes actuellement identifiées :
- **CIPI dataset** : dataset de piano classique avec partitions en MusicXML et niveaux de difficulté annotés ;
- **PianoLibrary** : base de morceaux classés par difficulté et par compositeur, potentiellement utile pour la recommandation ;
- datasets complémentaires de piano (par exemple MAESTRO, ASAP ou autres bases symboliques) si besoin pour enrichir les métadonnées musicales ou les informations de style.

Selon l’accessibilité réelle des données, le projet pourra s’appuyer soit sur un dataset principal unique, soit sur une combinaison de plusieurs sources.

## Première idée de workflow
1. Récupérer et nettoyer un dataset de morceaux de piano avec informations de difficulté.
2. Extraire ou utiliser des caractéristiques musicales pertinentes (compositeur, style, structure, informations symboliques, etc.).
3. Construire un modèle de machine learning pour prédire le niveau de difficulté d’un morceau.
4. Comparer plusieurs modèles de classification ou de régression selon la forme finale de la cible.
5. Développer une interface Streamlit permettant à l’utilisateur de sélectionner un style, un compositeur ou un morceau de référence, puis d’obtenir des recommandations adaptées à son niveau.

## Remarque
Le cœur du projet sera la prédiction de difficulté. La recommandation de morceaux similaires constituera une seconde couche applicative pour rendre le projet plus concret et utile.