"""CPU benchmark for steering throughput and nullspace direction construction."""
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from steer.engine import Steerer, apply, unit  # noqa: E402


def cpu_model():
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        if result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def synthetic_emb(seed=101, n=1200, dim=384):
    rng = np.random.default_rng(seed)
    supports = ((0, 1, 2, 3), (2, 3, 4, 5), (8, 9, 10, 11), (16, 17, 18, 19))
    emb = {}
    for i, concept in enumerate("ABCD"):
        emb[concept] = {"pos": {}, "neg": {}}
        for label, side in ((1, "pos"), (-1, "neg")):
            X = rng.normal(0, .65, (n, dim)).astype(np.float32)
            X[:, supports[i]] += label * 1.35
            emb[concept][side]["fit"] = unit(X).astype(np.float32)
    return emb


def main():
    rng = np.random.default_rng(2026)
    d = unit(rng.normal(size=384).astype(np.float32))
    results = {"system": {"cpu": cpu_model(), "python": platform.python_version(),
                          "numpy": np.__version__, "dtype": "float32", "dimension": 384},
               "apply": [], "nullspace_direction": {}}
    for n in (10_000, 100_000, 1_000_000):
        X = unit(rng.normal(size=(n, 384)).astype(np.float32)) if n < 1_000_000 else None
        times = []
        for _ in range(5):
            start = time.perf_counter()
            if n == 1_000_000:
                elapsed_apply = 0.0
                for offset in range(0, n, 100_000):
                    chunk = unit(rng.normal(size=(100_000, 384)).astype(np.float32))
                    chunk_start = time.perf_counter()
                    apply(chunk, d, .3)
                    elapsed_apply += time.perf_counter() - chunk_start
                times.append(elapsed_apply)
            else:
                apply(X, d, .3)
                times.append(time.perf_counter() - start)
        median = statistics.median(times)
        results["apply"].append({"vectors": n, "chunk_size": 100_000 if n == 1_000_000 else n,
                                 "repeat_seconds": times, "median_seconds": median,
                                 "microseconds_per_vector": median * 1e6 / n,
                                 "vectors_per_second": n / median})

    emb = synthetic_emb()
    direction_times = []
    for _ in range(5):
        start = time.perf_counter()
        st = Steerer(emb, seed=5)
        st.d("A", "nullspace")
        direction_times.append(time.perf_counter() - start)
    results["nullspace_direction"] = {"repeat_seconds": direction_times,
                                      "median_seconds": statistics.median(direction_times),
                                      "fit_vectors_per_side_per_concept": 1200}

    md = ["# CPU scale benchmark", "", f"- CPU: {results['system']['cpu']}",
          f"- Python: {results['system']['python']}", f"- NumPy: {results['system']['numpy']}",
          "- Data: synthetic float32, 384 dimensions; 1M measured in 100k chunks.", "",
          "## apply()", "", "| Vectors | Median total (s) | µs / vector | Vectors / s |",
          "|---:|---:|---:|---:|"]
    for row in results["apply"]:
        md.append(f"| {row['vectors']:,} | {row['median_seconds']:.4f} | {row['microseconds_per_vector']:.3f} | {row['vectors_per_second']:,.0f} |")
    ns = results["nullspace_direction"]
    md += ["", "## Nullspace direction", "", f"Steerer initialization and one direction lookup (5 runs, median): **{ns['median_seconds']:.4f} s**.", ""]
    (ROOT / "results" / "scale.md").write_text("\n".join(md), encoding="utf-8")
    (ROOT / "results" / "scale.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
