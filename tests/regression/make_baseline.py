"""Regenerate the regression baseline (only after a deliberate, documented change)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from test_regression import current, BASELINE  # noqa: E402

BASELINE.write_text(json.dumps(current(), indent=2))
print("wrote", BASELINE)
