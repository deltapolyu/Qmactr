# Qmactr

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run

Run one circuit:

```bash
PYTHONPATH=src python3 scripts/run_real.py \
  --circuits data/real/gf2^4_mult.qasm \
  --topologies data/topology_template.json \
  --mapping-episodes 180 \
  --transform-episodes 90
```

Run all bundled circuits:

```bash
PYTHONPATH=src python3 scripts/run_real.py \
  --topologies data/topology_template.json \
  --mapping-episodes 180 \
  --transform-episodes 90
```

Edit `data/topology_template.json` to define your own topology. To run multiple
topologies, pass comma-separated topology JSON paths.

Use GPU when available:

```bash
PYTHONPATH=src python3 scripts/run_real.py \
  --topologies data/topology_template.json \
  --device cuda
```

Use a larger budget:

```bash
PYTHONPATH=src python3 scripts/run_real.py \
  --topologies data/topology_template.json \
  --mapping-episodes 4000 \
  --transform-episodes 2000 \
  --device cuda
```

Results are written to `results/real` by default. Each run writes only `summary.json`
and the optimized `qmactr_*.qasm` files. Use `--out` to choose another output
directory.

```bash
PYTHONPATH=src python3 scripts/run_real.py \
  --circuits data/real/qft_20.qasm \
  --topologies data/topology_template.json \
  --out results/qft20_grid
```
