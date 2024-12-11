import os
import pathlib

PACKAGE_ROOT = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = PACKAGE_ROOT.parent.parent
