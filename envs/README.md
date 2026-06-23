# Runtime environments for `therapeutic_tools`

The tool computes features by shelling out to **two separate conda
environments** (see `runtime_envs.py`). Keep them separate — they pin different
PyTorch / CUDA / RDKit versions and will fight if merged.

| Env | Provides | Core library (single pip install) |
|-----|----------|-----------------------------------|
| `openrlhf` | RDKit descriptors, topology, structural alerts, pKa, ionization, logD | **MolGpKa** |
| `minimol`  | MiniMol + Ridge aqueous solubility | **MiniMol** |

Each core library declares its own heavy dependencies, so installing it pulls
the rest of the stack automatically:

- `molgpka` depends on `torch`, `torch-geometric`, `rdkit`, `scikit-learn`, `pandas`.
- `minimol` depends on `graphium`, which pulls `torch`, the PyG stack, and `rdkit`.

> Built/tested on Linux x86_64, Python 3.11. A CUDA GPU is expected for the
> original setup; CPU-only works too (see notes below), just slower.

---

## 1. `openrlhf` env — MolGpKa

```bash
conda create -p ./conda_env/openrlhf python=3.11 -y
conda activate ./conda_env/openrlhf

# MolGpKa has no PyPI release; install the pinned git commit.
# This single command also pulls torch + torch-geometric + rdkit + scikit-learn.
pip install "git+https://github.com/haydn-jones/MolGpKa.git@f23ebcb12bba7ea2c9295db9527ceb07188d600e"
```

Want a specific CUDA build of torch (the reference env used CUDA 13.0)? Install
torch first, then MolGpKa reuses it:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130   # or .../cpu
pip install "git+https://github.com/haydn-jones/MolGpKa.git@f23ebcb12bba7ea2c9295db9527ceb07188d600e"
```

---

## 2. `minimol` env — MiniMol

```bash
conda create -p ./conda_env/minimol python=3.11 -y
conda activate ./conda_env/minimol

# Pulls graphium + the torch/PyG stack + rdkit.
pip install minimol
```

The reference env used `torch 2.3.0 + CUDA 12.1`. MiniMol needs the compiled PyG
extensions (`torch_scatter` / `torch_sparse` / `torch_cluster`). If pip tries to
build those from source (slow) or fails, install torch + the matching prebuilt
wheels first, then MiniMol:

```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install minimol
# CPU-only: use the cpu torch index and https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

---

## 3. Point the code at your envs

`runtime_envs.py` reads the interpreter path from an env var, defaulting to the
hardcoded `/vast/...` path. Override both:

```bash
export THERAPEUTIC_TOOLS_OPENRLHF_PYTHON=/abs/path/conda_env/openrlhf/bin/python
export THERAPEUTIC_TOOLS_MINIMOL_PYTHON=/abs/path/conda_env/minimol/bin/python
```

## 4. Verify

```bash
python -m therapeutic_tools.doctor   # "runtimes" block must report both envs ok
```

This probes `rdkit`+`molgpka` in the openrlhf env and
`rdkit`+`minimol`+`graphium`+`torch_sparse` in the minimol env. Once both are
green, the tool can compute features on this machine — the DuckDB cache then
builds itself from scratch as molecules are queried.
