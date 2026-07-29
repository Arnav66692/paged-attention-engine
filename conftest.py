"""Put the repo root on sys.path so tests can import `engine` and `bench`."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
