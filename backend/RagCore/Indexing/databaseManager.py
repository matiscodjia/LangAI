import os
import re
import duckdb
import pandas as pd
import logging

from backend.RagCore.Indexing.metadataGenerator import MetadataGenerator
from backend.RagCore.Utils.configManager import ConfigManager
from backend.RagCore.Utils.pathProvider import PathProvider

# Setup logger
log = logging.getLogger("DuckDBManager")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class DuckDBManager:
    def __init__(self):
        """
        Initialize the metadata reader with the path to the DuckDB database.
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

    def text_file_to_duckdb(self, file_path: str) -> None:
        """
        Read a text file, extract metadata, and store it into DuckDB
        (if not already stored).
        """
        file_path = str(self.provider.raw_data(file_path))

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        match = re.search(r"(\d{4}-\d{2}-\d{2})", file_path)
        file_date = match.group(1) if match else "unknown"

        con = duckdb.connect(self.db_path)
        try:
            result = con.execute(
                "SELECT COUNT(*) FROM documents WHERE source = ?", [file_date]
            ).fetchone()[0]

            if result > 0:
                log.warning(f"Skipped: document '{file_date}' already in DuckDB.")
                con.close()
                return
        except duckdb.CatalogException:
            log.info("Table 'documents' does not exist yet — creating new one.")

        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        intro_text = "\n".join(full_text.splitlines()[:25])
        summary = ""
        global_theme = ""

        config_manager = ConfigManager()
        if config_manager.get_advanced_metadata():
            try:
                summary = self.metadata_gen.generate_summary(intro_text)
            except Exception as e:
                log.error(f"Error generating summary: {e}")

            try:
                global_theme = self.metadata_gen.generate_global_theme(full_text)
            except Exception as e:
                log.error(f"Error generating global theme: {e}")

        metadata = {
            "source": file_date,
            "date": file_date,
            "sommaire": summary,
            "theme_global": global_theme,
            "texte": full_text
        }

        df = pd.DataFrame([metadata])
        con.register("df", df)
        con.execute("INSERT INTO documents SELECT * FROM df")
        con.close()

        log.info(f"Document '{file_date}' added to DuckDB.")