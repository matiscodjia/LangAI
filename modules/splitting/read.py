import duckdb
import pandas as pd

def read_metadata_from_duckdb(duckdb_file: str) -> pd.DataFrame:
    """
    Se connecte à la base DuckDB et lit les métadonnées depuis la table 'documents'.

    :param duckdb_file: Chemin vers le fichier DuckDB.
    :return: DataFrame contenant les données lues.
    """
    # Connexion à DuckDB
    con = duckdb.connect(duckdb_file)
    # Exécute une requête SQL pour récupérer toutes les lignes de la table 'documents'
    df = con.execute("SELECT * FROM documents").fetchdf()
    con.close()
    return df

if __name__ == "__main__":
    duckdb_file = "metadata.duckdb"  # chemin vers votre base DuckDB
    df_metadata = read_metadata_from_duckdb(duckdb_file)
    print("Données lues depuis DuckDB:")
    print(df_metadata)