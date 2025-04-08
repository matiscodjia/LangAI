from modules.data_managing.read import *
from modules.data_managing.metadata_gen import *
from modules.embedding.store import *
from modules.splitting.splitter import *

## Il s'agit ici de mettre au point l'état initial de nos bases de données
## (Base de texte et base de vecteurs)
## Quel(s) texte à analyser et stocké en BD
## Il faut choisir, quelle méthode de splitting à mettre en oeuvre etc.

###Charger les fichiers dans la base DuckDB (à réaliser une seule fois - offline)
#Trouver la liste des fichiers à charger en db
RAW_DATA_PATH="/Users/mtis/Local/Code/GitRepos/LangAI/data/raw_data"
DUCKDB_PATH = "/Users/mtis/Local/Code/GitRepos/LangAI/data/metadata.duckdb"
CHROMA_PATH = "/Users/mtis/Local/Code/GitRepos/LangAI/data/chroma_db"
files = make_file_list(RAW_DATA_PATH)
print(files)

#On peut choisir d'échantillonner les documents en fonction des besoins.

#Parcourir chaque fichier et en extraire les informations dans les méta données

for key, value in files.items():
   process_text_file_to_duckdb(value,DUCKDB_PATH)
#Pour chaque document, splitter suivant les stratégies choisies (une seule fois -offline aussi)

docs_semantic = semantic_split()
docs_recursive = recursive_split()
docs_fixed = base_split()

#Vectoriser chaque groupe de documents dans une collection de la base
#On pourra alors filtrer la récupération sur une collection en particulier (une seule fois)

store_documents(docs=docs_semantic, collection_name="semantic",path=CHROMA_PATH)
store_documents(docs=docs_recursive, collection_name="recursive",path=CHROMA_PATH)
store_documents(docs=docs_fixed, collection_name="fixed",path=CHROMA_PATH)
