"""
Author: L. Saetta
Date last modified: 2026-09-02
License: MIT
Description: Configure the repository root for local test imports.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
