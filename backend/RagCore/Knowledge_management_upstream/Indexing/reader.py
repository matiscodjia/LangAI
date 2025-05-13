import duckdb
from collections import defaultdict
import os
import random
from reader import resolve_path
def read_metadata_from_duckdb(duckdb_file):
    duckdb_file = str(resolve_path(duckdb_file))
    """
    Se connecte à la base DuckDB et lit les métadonnées depuis la table 'documents'.

    :param duckdb_file: Chemin vers le fichier DuckDB.
    :return: DataFrame contenant les données lues.
    """

    # Connexion à DuckDB
    con = duckdb.connect(duckdb_file)
    df = con.execute("SELECT * FROM documents").fetchdf()
    con.close()
    return df

def make_file_list(path):
    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.txt'): 
                list_of_files[filename.split('.')[0]] = os.sep.join([dirpath, filename])
    return list_of_files

def sample_file_list_by_month(file_dict, n=1):
    grouped = defaultdict(list)

    for key, path in file_dict.items():
        try:
            month_key = key[:7]  
            grouped[month_key].append((key, path))
        except:
            continue  

   
    sampled = {}
    for month, files in grouped.items():
        selected = random.sample(files, min(n, len(files)))
        for key, path in selected:
            sampled[key] = path

    return sampled
