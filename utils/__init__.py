from pathlib import Path
from .mytqdm import *
PROJECT_PATH = str(Path(__file__).parents[1])  # absolute path

DATA_PATH = str(Path(PROJECT_PATH) / ".data")