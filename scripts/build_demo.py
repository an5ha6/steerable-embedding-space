"""Prebuild the demo state so `streamlit run app.py` opens in seconds. Run: python scripts/build_demo.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steer import demo_state  # noqa: E402

for name in ("core", "paper_space"):
    t = time.time()
    getattr(demo_state, name)()
    print(f"{name}: ready in {time.time() - t:.0f}s")
