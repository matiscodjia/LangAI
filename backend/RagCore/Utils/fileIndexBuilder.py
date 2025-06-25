import os
import random
import logging
from pathlib import Path
from collections import defaultdict

from backend.RagCore.Utils.configManager import ConfigManager
from backend.RagCore.Utils.pathProvider import PathProvider

# Setup logger
log = logging.getLogger("FileIndexBuilder")
logging.basicConfig(level=logging.INFO, format="%(message)s")

class FileIndexBuilder:
    def __init__(self):
        path_provider = PathProvider()
        config_manager = ConfigManager()
        root_dir = config_manager.get_raw_data_path()
        root_dir = path_provider.raw_data(root_dir)
        self.root_dir = root_dir
        self.file_index = {}

    def build_index(self) -> dict:
        log.info(f"Scanning files in: {self.root_dir}")
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith(".txt"):
                    key = Path(filename).stem
                    self.file_index[key] = os.path.join(dirpath, filename)

        log.info(f"{len(self.file_index)} text file(s) indexed.")
        return self.file_index

    def sample_by_month(self, n: int = 1) -> dict:
        if not self.file_index:
            self.build_index()

        grouped = defaultdict(list)
        for key, path in self.file_index.items():
            try:
                month_key = key[:7]
                grouped[month_key].append((key, path))
            except Exception:
                continue

        sampled = {}
        for month, files in grouped.items():
            selected = random.sample(files, min(n, len(files)))
            for key, path in selected:
                sampled[key] = path

        log.info(f"Sampled {len(sampled)} file(s) from {len(grouped)} months.")
        return sampled