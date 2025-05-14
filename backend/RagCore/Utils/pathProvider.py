from pathlib import Path
import duckdb


class PathProvider:
    def __init__(self, root: Path = None):
        """
        Initialize with a project root. If not provided, autodetect based on pyproject.toml.
        """
        self.root = root or self._detect_project_root()

    def _detect_project_root(self) -> Path:
        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                return parent
        raise FileNotFoundError("Could not locate project root (pyproject.toml not found).")

    def data(self, filename: str = "") -> Path:
        base = self.root / "data"
        return base / filename if filename else base

    def raw_data(self, filename: str = "") -> Path:
        base = self.data("raw_data")
        return base / filename if filename else base

    def metadatas(self, filename: str = "") -> Path:
        base = self.data("metadatas")
        return base / filename if filename else base

    def chroma(self, filename: str = "") -> Path:
        base = self.data("chromaDB")
        return base / filename if filename else base

    def frontend(self, filename: str = "") -> Path:
        base = self.root / "frontend"
        return base / filename if filename else base

    def metadata_db(self, init_if_missing: bool = True) -> Path:
        db_path = self.data("metadata.duckdb")

        if db_path.exists():
            try:
                duckdb.connect(str(db_path)).close()
            except duckdb.IOException:
                print(f"[Warning] Invalid DuckDB file at {db_path}. Recreating it.")
                db_path.unlink()

        if init_if_missing and not db_path.exists():
            print(f"[Init] Creating new DuckDB database at {db_path}")
            con = duckdb.connect(str(db_path))
            con.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    source TEXT,
                    date TEXT,
                    sommaire TEXT,
                    theme_global TEXT,
                    texte TEXT,
                    is_already_splitted BOOLEAN DEFAULT FALSE
                )
            """)
            con.close()

        return db_path