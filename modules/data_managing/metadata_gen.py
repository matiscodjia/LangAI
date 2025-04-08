import json
import os
from langchain_ollama.llms import OllamaLLM
from langchain_community.llms import LlamaCpp
import pandas as pd
import spacy
import logging
import re
import duckdb

from pathlib import Path

def find_project_root(marker="pyproject.toml") -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root with marker '{marker}' not found.")

def resolve_path(relative_path: str) -> str:
    root = find_project_root()
    return str((root / relative_path).resolve())
#model_mistral = LlamaCpp(
#   model_path="/Users/mtis/Local/Code/GitRepos/llama.cpp/llama.cpp/models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#    n_ctx = 32768,
#    temperature=0.5,
#    n_threads=8
#)
model_llama = OllamaLLM(model="llama3.2")
chain = model_llama

# Charger un modèle pour le NER (ici en français)
nlp = spacy.load("fr_core_news_sm")

def call_llm_summary(text: str) -> str:
    # Utilise le llm pour générer le sommaire
    print("Génération du sommaire")
    prompt_str = (
        "Donne-moi uniquement la liste des points du sommaire, sans introduction, sans formule ou modalisateurs.\n\n"
        "Texte:\n" + text
    )
    try:
        response = chain.invoke(prompt_str)
        return response
    except Exception as e:
        logging.error("Erreur lors de l'appel du LLM pour le sommaire: %s", e)
        return "No data"

def call_llm_global_theme(text: str) -> str:
    # Utilise le llm pour générer le topic
    print("Génération du thème du document")
    prompt_str = (
        "Donne moi le contexte global en 2 phrases maximum, sans introduction, sans formule juste de l'info brute\n\n"
        "Texte:\n" + text
    )
    try:
        response = chain.invoke(prompt_str)
        return response
    except Exception as e:
        logging.error("Erreur lors de l'appel du LLM pour le thème global: %s", e)
        return "No data"

def extract_named_entities(text: str) -> list:
    # Extraction des entités nommées via spaCy
    print("Extraction des entités nommées")
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({"entite": ent.text, "label": ent.label_})
    return entities

def process_text_file_to_duckdb(file_path: str, duckdb_file):
    """
    Lit un fichier texte, extrait les métadonnées (sommaire, entités nommées, thème global)
    et stocke le tout dans une base de données DuckDB.
    """
    file_path = str(resolve_path(file_path))
    duckdb_file = str(resolve_path(duckdb_file))
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    # Lecture du fichier texte complet
    with open(file_path, "r", encoding="utf-8") as f:
        text_complet = f.read()
    
    # Extraction de la date depuis le chemin (format "YYYY-MM-DD")
    match = re.search(r'(\d{4}-\d{2}-\d{2})', file_path)
    file_date = match.group(1) if match else "unknown"
    
    # Pour générer le sommaire, on utilise les 25 premières lignes
    text_debut = "\n".join(text_complet.splitlines())
    # Extraction des entités nommées via spaCy (limitées aux 10 premières)
    entites = extract_named_entities(text_complet)[:10]
    # Appels aux fonctions LLM pour générer le sommaire et le thème global
    sommaire = call_llm_summary(text_debut)
    theme_global = call_llm_global_theme(text_complet)
    # Préparer un dictionnaire de métadonnées
    metadata = {
        "source": file_date,
        "date": file_date,
        "sommaire": sommaire,
        "theme_global": theme_global,
        "entites_nominees": json.dumps(entites),  # stockage sous forme de chaîne JSON
        "texte":text_complet
    }
    # Conversion en DataFrame
    df = pd.DataFrame([metadata])
    # Connexion à DuckDB et stockage dans une table "documents"
    if duckdb_file:

        con = duckdb.connect(duckdb_file)
        # Créer la table si elle n'existe pas (structure basée sur le DataFrame)
        con.execute("CREATE TABLE IF NOT EXISTS documents AS SELECT * FROM df WHERE 1=0")
        # Insertion des données dans la table
        con.execute("INSERT INTO documents SELECT * FROM df")
        con.close()
        print(f"Les métadonnées ont été extraites et stockées dans la base DuckDB : {duckdb_file}")
    else:
        print("Chemin du fichier inexistant")
if __name__ == "__main__":
    process_text_file_to_duckdb("data/raw_data/1881-01-11.txt","data/metadata.duckdb")
