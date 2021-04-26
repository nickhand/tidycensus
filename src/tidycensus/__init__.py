from pathlib import Path

__version__ = "0.0.1"

DATA_DIR = Path(__file__).parent.absolute() / "data"

from .acs import get_acs
