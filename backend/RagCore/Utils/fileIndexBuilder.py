import os
import random
from pathlib import Path
from collections import defaultdict


class FileIndexBuilder:
    def __init__(self, root_dir: Path):
        """
        Initialize the builder with a root directory containing text files.
        :param root_dir: Path to the directory to scan
        """
        self.root_dir = root_dir
        self.file_index = {}

    def build_index(self) -> dict:
        """
        Recursively scans the root directory and builds an index of .txt files.
        Keys are based on the filename stem (e.g. '1881-01-11' from '1881-01-11.txt').
        :return: Dict[str, str]
        """
        print(f"Scanning files in: {self.root_dir}")
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith(".txt"):
                    key = Path(filename).stem
                    self.file_index[key] = os.path.join(dirpath, filename)

        print(f"{len(self.file_index)} text files indexed.")
        return self.file_index

    def sample_by_month(self, n: int = 1) -> dict:
        """
        Samples up to `n` files per month from the current file index.
        Assumes keys are in 'YYYY-MM-DD' format.
        :param n: Number of samples per month
        :return: Dict[str, str] of sampled files
        """
        if not self.file_index:
            self.build_index()

        grouped = defaultdict(list)
        for key, path in self.file_index.items():
            try:
                month_key = key[:7]  # Extract YYYY-MM
                grouped[month_key].append((key, path))
            except Exception:
                continue

        sampled = {}
        for month, files in grouped.items():
            selected = random.sample(files, min(n, len(files)))
            for key, path in selected:
                sampled[key] = path

        print(f"Sampled {len(sampled)} files from {len(grouped)} months.")
        return sampled