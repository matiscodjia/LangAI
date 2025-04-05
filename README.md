# LangAI

LangAI est un projet visant à analyser et traiter des documents historiques en combinant différentes techniques de prétraitement, de splitting, d'embedding et de retrieval. Le pipeline intègre plusieurs stratégies de splitting (sémantique, récursif, basé sur les tokens) afin de comparer l’impact de chaque méthode sur la qualité des réponses dans une approche de Retrieval-Augmented Generation (RAG).

---

## Table des matières

- [Architecture du projet](#architecture-du-projet)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Modules principaux](#modules-principaux)
  - [Data Processor](#data-processor)
  - [Splitting](#splitting)
  - [Embedding](#embedding)
  - [Pipeline](#pipeline)
- [Utilisation](#utilisation)
- [Configuration et dépendances](#configuration-et-dépendances)
- [Contribuer](#contribuer)
- [Licence](#licence)

---

## Architecture du projet

LangAI est structuré en plusieurs modules, chacun correspondant à une étape clé du pipeline :

1. **Data Processor**  
   - Extraction et préparation des métadonnées à partir de documents bruts.
   - Lecture des données depuis DuckDB et création de DataFrames.

2. **Splitting**  
   - **SemanticChunker** : Utilise des embeddings pour regrouper des passages sémantiquement cohérents.
   - **RecursiveCharacterTextSplitter** : Découpe le texte selon des séparateurs naturels (paragraphes, lignes) avec chevauchement.
   - **CharacterTextSplitter (tiktoken)** : Découpe le texte selon le nombre de tokens via l'encodeur `cl100k_base`.

3. **Embedding**  
   - Calcul des embeddings des documents/chunks avec OllamaEmbeddings (modèle "nomic-embed-text").
   - Stockage des embeddings et des documents dans ChromaDB avec des namespaces spécifiques.

4. **Pipeline**  
   - Orchestration globale du traitement, depuis la lecture des données jusqu'au stockage final pour le retrieval.

---

## Installation

### Prérequis

- Python 3.11 ou supérieur
- pip

### Installation des dépendances

Exécutez la commande suivante à la racine du projet :

```bash
pip install -r requirements.txt
```


Data Processor
	•	metadata_gen.py :
Lit les fichiers texte, extrait les métadonnées (sommaire, thème global, entités nommées via spaCy), et stocke ces informations dans une base DuckDB.

Splitting
	•	splitter.py :
Contient des fonctions pour découper les textes selon plusieurs méthodes (séquentielle et parallèle via multiprocessing).
	•	read.py :
Fournit des fonctions pour lire les données depuis DuckDB.

Embedding
	•	store.py :
Calcule les embeddings des documents via OllamaEmbeddings et les insère dans ChromaDB dans des namespaces spécifiques, facilitant ainsi la comparaison des méthodes de splitting.