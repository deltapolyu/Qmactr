from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_TOPOLOGIES = "data/topology_template.json"
DEFAULT_MAPPING_EPISODES = 5000
DEFAULT_TRANSFORM_EPISODES = 2000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Qmactr on OpenQASM circuits."
    )
    parser.add_argument(
        "--root",
        default="data/real",
        help="Directory scanned when --circuits is omitted.",
    )
    parser.add_argument(
        "--circuits",
        default="",
        help="Comma-separated QASM paths. Empty means scan --root.",
    )
    parser.add_argument(
        "--topologies",
        default=DEFAULT_TOPOLOGIES,
        help="Comma-separated topology JSON file paths.",
    )
    parser.add_argument(
        "--mapping-episodes", type=int, default=DEFAULT_MAPPING_EPISODES
    )
    parser.add_argument(
        "--transform-episodes", type=int, default=DEFAULT_TRANSFORM_EPISODES
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/real")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training/inference device: auto, cpu, cuda, cuda:0, ...",
    )
    args = parser.parse_args()

    from qmactr.evaluate import evaluate_suite

    if args.circuits.strip():
        circuits = [x.strip() for x in args.circuits.split(",") if x.strip()]
    else:
        root = Path(args.root)
        circuits = sorted(str(p) for p in root.glob("*.qasm"))
    if not circuits:
        raise SystemExit("No .qasm files found. Check --root or --circuits.")

    topologies = [x.strip() for x in str(args.topologies).split(",") if x.strip()]

    results = evaluate_suite(
        circuits=circuits,
        topologies=topologies,
        out_dir=Path(args.out),
        seed=args.seed,
        mapping_episodes=args.mapping_episodes,
        transform_episodes=args.transform_episodes,
        device=args.device,
    )

    print("Saved results to", args.out)
    for r in results:
        print(
            f"{r.benchmark:24s} {r.topology:6s} "
            f"qmactr={r.qmactr:8.3f} "
            f"off_same_model={r.qmactr_off_same_model:8.3f} "
            f"transform_gain={r.transform_gain_vs_off_same_model_pct:6.2f}%"
        )


if __name__ == "__main__":
    main()
