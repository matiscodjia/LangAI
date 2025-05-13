import os
import random
import re
import duckdb
from collections import defaultdict
from pathlib import Path
import pandas as pd

from RagCore.KnowledgeManagement.Indexing.metadataGenerator import  MetadataGenerator
from RagCore.Utils.pathProvider import PathProvider


class DuckDBManager:
    def __init__(self):
        """
        Initialize the metadata reader with the path to the DuckDB database.
        If not provided, it uses the default project metadata path.
        """
        self.provider = PathProvider()
        self.db_path = self.provider.metadata_db()
        self.metadata_gen = MetadataGenerator()
        

    def read_metadata(self) -> pd.DataFrame:
        """
        Connects to the DuckDB database and reads the 'documents' table.
        :return: DataFrame with the metadata.
        """
        con = duckdb.connect(str(self.db_path))
        df = con.execute("SELECT * FROM documents").fetchdf()
        con.close()
        return df
    def process_text_file_to_duckdb(self, file_path: str):
        """
        Read a text file, extract metadata (outline, named entities, global theme),
        and store everything into a DuckDB database.
        """
        
        file_path = str(self.provider.raw_data(file_path))
        duckdb_file = self.db_path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Read the entire content of the text file
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # Extract date from filename if possible
        match = re.search(r'(\d{4}-\d{2}-\d{2})', file_path)
        file_date = match.group(1) if match else "unknown"

        # Use the first 25 lines to generate the summary
        intro_text = "\n".join(full_text.splitlines()[:25])

        # LLM calls
        summary = self.metadata_gen.generate_summary(intro_text)
        global_theme = self.metadata_gen.generate_global_theme(full_text)

        # Build metadata dictionary
        metadata = {
            "source": file_date,
            "date": file_date,
            "sommaire": summary,
            "theme_global": global_theme,
            "texte": full_text
        }

        # Convert to DataFrame
        df = pd.DataFrame([metadata])

        # Store to DuckDB
        if duckdb_file:
            con = duckdb.connect(duckdb_file)
            con.execute("CREATE TABLE IF NOT EXISTS documents AS SELECT * FROM df WHERE 1=0")
            con.execute("INSERT INTO documents SELECT * FROM df")
            con.close()
            print(f"Metadata extracted and stored in DuckDB: {duckdb_file}")
        else:
            print("DuckDB file path is missing.")

    
