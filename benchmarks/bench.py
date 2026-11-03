"""Reproducible performance benchmark.

    python benchmarks/bench.py [--events 50000]

Measures generation and full-chain throughput, peak Python-heap memory and fit
time for each resonance, and writes outputs/benchmarks.json together with the
environment record.
"""
import argparse
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from particlelab.analysis import simulate, fit_peak  # noqa: E402
from particlelab.analysis.compare import SIM_FIT  # noqa: E402
from particlelab.domain.particles import RESONANCES  # noqa: E402
from particlelab.physics import generate  # noqa: E402
from particlelab.provenance import environment, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=50_000)
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    out = {"environment": environment(), "events": a.events, "results": {}}
    for name in RESONANCES:
        gen, full, mem = [], [], 0.0
        for r in range(a.repeat):
            t = time.perf_counter(); generate(name, a.events, seed=r); gen.append(time.perf_counter() - t)
            tracemalloc.start()
            t = time.perf_counter(); sim = simulate(name, a.events, seed=r); full.append(time.perf_counter() - t)
            mem = max(mem, tracemalloc.get_traced_memory()[1] / 1e6)
            tracemalloc.stop()
        t = time.perf_counter(); fit_peak(sim.reco_mass, name, background=False, **SIM_FIT[name]); tf = time.perf_counter() - t
        res = {"generate_events_per_s": a.events / min(gen), "full_chain_events_per_s": a.events / min(full),
               "selected_fraction": sim.acceptance, "peak_memory_MB": mem, "fit_seconds": tf}
        out["results"][name] = res
        print(name, {k: round(v, 3) for k, v in res.items()}, flush=True)
    save_json(out, ROOT / "outputs" / "benchmarks.json")


if __name__ == "__main__":
    main()
