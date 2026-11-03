# particlelab_fast

This package is an optional compiled version of the two slowest steps of the simulation. Both steps take about half the run time each:

- `simulate_hits`: helix crossings, multiple scattering, smearing and efficiency.
- `fit_tracks`: weighted circle fit and s–z line fit.

Each function is a line-by-line port of the NumPy code in `particlelab/detector/`. NumPy still draws every random number and passes it in. For the same seed, the two versions therefore give the same hits and fits up to floating-point rounding. `tests/unit/test_fast_backend.py` checks this.

Tracks are processed in parallel on all cores. Set `RAYON_NUM_THREADS` to limit the number of threads.

## Build and install

You need Rust (https://rustup.rs) and maturin (`pip install maturin`).

```bash
maturin build --release
```

```bash
pip install --force-reinstall target/wheels/particlelab_fast-*.whl
```

On Windows without Visual Studio, build with the GNU toolchain. Install it with `rustup-init --default-host x86_64-pc-windows-gnu --profile minimal`, which needs about 0.9 GB, then build:

```bash
maturin build --release --target x86_64-pc-windows-gnu
```

## Using it

`particlelab` uses the package automatically when it is installed and its `API_VERSION` matches. To run in NumPy instead, set `PARTICLELAB_BACKEND=numpy` or call `particlelab.detector._backend.use("numpy")`. Each run record stores the backend that was used.
