from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import re

from .circuit import Gate, QuantumCircuit


_QREG_RE = re.compile(r"qreg\s+([A-Za-z_]\w*)\[(\d+)\]\s*$")
_QUBIT_RE = re.compile(r"qubit\[(\d+)\]\s+([A-Za-z_]\w*)\s*$")
_ARG_RE = re.compile(r"([A-Za-z_]\w*)\[(\d+)\]\s*$")


def _safe_eval_expr(expr: str) -> float:

    expr = expr.strip().replace("^", "**")
    allowed = {"pi": math.pi}
    return float(eval(expr, {"__builtins__": {}}, allowed))


def _strip_comments(text: str) -> str:
    lines: List[str] = []
    for line in text.splitlines():
        base = line.split("//", 1)[0].strip()
        if base:
            lines.append(base)
    return "\n".join(lines)


def _split_statements(text: str) -> List[str]:
    parts = [p.strip() for p in text.split(";")]
    return [p for p in parts if p]


def _registers(statements: List[str]) -> Dict[str, Tuple[int, int]]:
    regs: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for st in statements:
        m1 = _QREG_RE.match(st)
        if m1:
            name, size_s = m1.group(1), m1.group(2)
            size = int(size_s)
            regs[name] = (offset, size)
            offset += size
            continue
        m2 = _QUBIT_RE.match(st)
        if m2:
            size_s, name = m2.group(1), m2.group(2)
            size = int(size_s)
            regs[name] = (offset, size)
            offset += size
    return regs


def _resolve_arg(arg: str, regs: Dict[str, Tuple[int, int]]) -> Optional[int]:
    m = _ARG_RE.match(arg.strip())
    if not m:
        return None
    name, idx_s = m.group(1), m.group(2)
    if name not in regs:
        return None
    base, size = regs[name]
    idx = int(idx_s)
    if idx < 0 or idx >= size:
        return None
    return base + idx


def load_qasm_circuit(path: str | Path, strict: bool = False) -> QuantumCircuit:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    statements = _split_statements(_strip_comments(text))
    regs = _registers(statements)

    gates: List[Gate] = []
    max_qubit_seen = -1
    skipped_ops = 0

    for st in statements:
        low = st.lower()
        if (
            low.startswith("openqasm")
            or low.startswith("include")
            or low.startswith("qreg ")
            or low.startswith("qubit[")
            or low.startswith("creg ")
            or low.startswith("bit[")
            or low.startswith("measure ")
            or low.startswith("barrier ")
            or low.startswith("reset ")
        ):
            continue

        m = re.match(r"([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s+(.+)$", st)
        if not m:
            skipped_ops += 1
            continue

        op = m.group(1).lower()
        param_s = m.group(2)
        arg_s = m.group(3)
        args = [a.strip() for a in arg_s.split(",") if a.strip()]
        qs = [_resolve_arg(a, regs) for a in args]
        if any(q is None for q in qs):
            skipped_ops += 1
            continue
        q_ints = [int(q) for q in qs if q is not None]
        if q_ints:
            max_qubit_seen = max(max_qubit_seen, max(q_ints))

        if op in {"cx", "cnot"} and len(q_ints) == 2:
            gates.append(Gate("cx", q_ints[0], q_ints[1]))
            continue
        if op in {"x", "h", "sx"} and len(q_ints) == 1:
            gates.append(Gate(op, q_ints[0]))
            continue
        if op == "u" and len(q_ints) == 1 and (param_s is None or not param_s.strip()):

            gates.append(Gate("sx", q_ints[0]))
            continue
        if op in {"rz", "p", "u1"} and len(q_ints) == 1:
            theta = _safe_eval_expr(param_s) if param_s is not None else 0.0
            gates.append(Gate("rz", q_ints[0], None, theta))
            continue

        if len(q_ints) == 1 and op in {"z", "s", "sdg", "t", "tdg"}:
            phase = {
                "z": math.pi,
                "s": math.pi / 2.0,
                "sdg": -math.pi / 2.0,
                "t": math.pi / 4.0,
                "tdg": -math.pi / 4.0,
            }[op]
            gates.append(Gate("rz", q_ints[0], None, phase))
            continue

        if len(q_ints) == 2 and op in {"cz", "cp", "ecr", "rxx", "ryy", "rzz"}:
            gates.append(Gate("cx", q_ints[0], q_ints[1]))
            continue
        if len(q_ints) == 2 and op == "swap":
            gates.append(Gate("cx", q_ints[0], q_ints[1]))
            gates.append(Gate("cx", q_ints[1], q_ints[0]))
            gates.append(Gate("cx", q_ints[0], q_ints[1]))
            continue

        if len(q_ints) == 1:
            theta = _safe_eval_expr(param_s.split(",")[0]) if param_s else None
            gates.append(Gate("u", q_ints[0], None, theta))
            continue

        skipped_ops += 1

    if not regs and max_qubit_seen >= 0:
        num_qubits = max_qubit_seen + 1
    else:
        num_qubits = sum(size for _, size in regs.values())

    if strict and skipped_ops > 0:
        raise ValueError(f"Unsupported statements in QASM: {skipped_ops}")
    if num_qubits <= 0:
        raise ValueError(f"No qubits parsed from QASM file: {p}")

    return QuantumCircuit(num_qubits=num_qubits, gates=gates)


def _fmt_angle(theta: float) -> str:

    return format(float(theta), ".12g")


def dump_qasm_circuit(
    circuit: QuantumCircuit,
    path: str | Path,
    include_header: bool = True,
) -> Path:
    p = Path(path)
    lines: List[str] = []
    if include_header:
        lines.append("OPENQASM 2.0;")
        lines.append('include "qelib1.inc";')
    lines.append(f"qreg q[{circuit.num_qubits}];")
    lines.append("")

    for g in circuit.gates:
        op = g.op.lower()
        if g.q1 is not None:
            lines.append(f"{op} q[{g.q0}],q[{g.q1}];")
            continue
        if op == "u" and g.param is None:
            lines.append(f"sx q[{g.q0}];")
            continue
        if op == "u" and g.param is not None:
            lines.append(f"u({_fmt_angle(g.param)},0,0) q[{g.q0}];")
            continue
        if g.param is not None:
            lines.append(f"{op}({_fmt_angle(g.param)}) q[{g.q0}];")
        else:
            lines.append(f"{op} q[{g.q0}];")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
