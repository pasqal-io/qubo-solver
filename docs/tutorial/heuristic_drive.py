# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional, List

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import torch

# -------------------------------
# QUBO Solver imports
# -------------------------------
from qubosolver.qubo_instance import QUBOInstance
from qubosolver.saveload import load_qubo_dataset
from qubosolver.qubo_types import DriveType
from qubosolver.solver import QuboSolver
from qubosolver.config import (
    SolverConfig,
    EmbeddingConfig,
    DriveShapingConfig,
    LocalEmulator,
)

# -------------------------------
# Devices / backends
# -------------------------------
try:
    from qoolqit.devices import DigitalAnalogDevice
except Exception:
    from qoolqit.devices import DigitalAnalogDevice  # fallback (selon ton env)

try:
    from pulser_simulation import QutipBackendV2
except Exception:
    QutipBackendV2 = None

# EMU backends (si dispo chez toi)
try:
    from emu_sv import SVBackend
except Exception:
    SVBackend = None

try:
    from emu_mps import MPSBackend
except Exception:
    MPSBackend = None


# ============================================================
# User config
# ============================================================

DATASET_DIR = Path("/home/ynaghmouchi/qubo-solver/qubosolver_logs/tutorial/01-dataset-generation-and-loading")  # <-- adapte si besoin
CSV_PATH = Path("results_adiabatic_vs_heuristic.csv")

# Ta logique: greedy_traps = max_atom_num
DA_DEVICE = DigitalAnalogDevice()

# Parallélisation
MAX_WORKERS_CAP = 64  # mets 8/16/32 selon ta machine
MP_CONTEXT = "spawn"  # plus safe

# ============================================================
# Helpers
# ============================================================

def load_datasets_by_size(directory: Path) -> Dict[int, Any]:
    """
    Charge les fichiers raw_qubo_dataset_size_{size}.pt
    """
    pattern = re.compile(r"raw_qubo_dataset_size_(\d+)\.pt$")
    datasets_by_size: Dict[int, Any] = {}
    directory.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(directory):
        m = pattern.match(filename)
        if not m:
            continue
        size = int(m.group(1))
        ds = load_qubo_dataset(str(directory / filename))
        datasets_by_size[size] = ds
        print(f"Loaded dataset size={size} from {directory/filename}")

    return dict(sorted(datasets_by_size.items(), key=lambda kv: kv[0]))


def pick_backend_by_size(n: int):
    """
    Backend dépendant de la taille (même idée que ton script d’étude).
    Ajuste les seuils si tu veux.
    """
    if n <= 15 and QutipBackendV2 is not None:
        return LocalEmulator(backend_type=QutipBackendV2), "qutip_v2"

    if 16 <= n <= 23 and SVBackend is not None:
        return LocalEmulator(backend_type=SVBackend), "emu_sv"

    if n >= 24 and MPSBackend is not None:
        return LocalEmulator(backend_type=MPSBackend), "emu_mps"

    # fallback si EMU pas dispo
    if SVBackend is not None:
        return LocalEmulator(backend_type=SVBackend), "emu_sv_fallback"

    # dernier recours: qutip si dispo
    if QutipBackendV2 is not None:
        return LocalEmulator(backend_type=QutipBackendV2), "qutip_fallback"

    raise RuntimeError("No backend available (QutipBackendV2 / SVBackend / MPSBackend).")


def make_quantum_config(n: int, drive_type: DriveType) -> SolverConfig:
    embedding_cfg = EmbeddingConfig(
        embedding_method="greedy",
        greedy_traps=DA_DEVICE._device.max_atom_num,
        # pas de spacing ici (tu as demandé “config naturelle”)
    )

    backend, backend_name = pick_backend_by_size(n)

    cfg = SolverConfig(
        use_quantum=True,
        device=DA_DEVICE,
        embedding=embedding_cfg,
        drive_shaping=DriveShapingConfig(drive_shaping_method=drive_type),
        backend=backend,
    )
    return cfg


def make_cplex_config() -> SolverConfig:
    # Selon ton setup, tu as peut-être ClassicalConfig etc.
    # Ici, on reste minimal: si ton qubosolver “classique cplex” se configure autrement,
    # remplace cette fonction par TA config habituelle.
    from qubosolver.config import ClassicalConfig  # si existe chez toi

    return SolverConfig(
        use_quantum=False,
        classical=ClassicalConfig(classical_solver_type="cplex"),
    )


def best_cost_and_prob(sol) -> Tuple[Optional[float], Optional[float]]:
    """
    Retourne (best_cost, prob(best_cost)).
    prob(best_cost) = somme des probabilités des bitstrings qui atteignent le coût minimal.
    """
    if sol is None or sol.costs is None:
        return None, None
    if not isinstance(sol.costs, torch.Tensor) or sol.costs.numel() == 0:
        return None, None

    costs = sol.costs
    best = float(torch.min(costs).item())

    if getattr(sol, "probabilities", None) is None:
        # si pas de probas, on peut approx via counts
        if getattr(sol, "counts", None) is None or not isinstance(sol.counts, torch.Tensor):
            return best, None
        probs = sol.counts.float() / sol.counts.sum()
    else:
        probs = sol.probabilities

    mask = (costs == torch.min(costs))
    prob_best = float(probs[mask].sum().item())
    return best, prob_best


def run_solver(instance: QUBOInstance, cfg: SolverConfig) -> Any:
    solver = QuboSolver(instance, cfg)
    return solver.solve()


# ============================================================
# CPLEX baseline (séquentiel)
# ============================================================

def compute_cplex_baseline(datasets: Dict[int, Any]) -> Dict[Tuple[int, int], float]:
    """
    key = (size, idx) -> best_cplex_cost
    """
    baseline: Dict[Tuple[int, int], float] = {}
    cfg = make_cplex_config()

    print("\n=== Phase 1: CPLEX baseline (sequential) ===")
    for size, ds in datasets.items():
        for idx in range(len(ds)):
            Q, _ = ds[idx]  # (coeffs, solution=None)
            Q = Q.to(dtype=torch.float32, device="cpu")

            t0 = time.perf_counter()
            try:
                sol = run_solver(QUBOInstance(coefficients=Q), cfg)
                best, _ = best_cost_and_prob(sol)
                if best is None:
                    raise RuntimeError("No best cost from CPLEX solve")
                baseline[(size, idx)] = float(best)
                dt = time.perf_counter() - t0
                print(f"[CPLEX] size={size} idx={idx} best={best} time={dt:.2f}s")
            except Exception as e:
                dt = time.perf_counter() - t0
                print(f"[CPLEX][FAIL] size={size} idx={idx} time={dt:.2f}s err={e}")
                traceback.print_exc()
                baseline[(size, idx)] = float("nan")

    return baseline


# ============================================================
# Worker quantum task
# ============================================================

def solve_quantum_task(task: Tuple[int, int, torch.Tensor, str]) -> Dict[str, Any]:
    """
    task = (size, idx, Q, mode)  where mode in {"adiabatic", "heuristic"}
    """
    size, idx, Q, mode = task

    # limiter threads par process (stabilité + perf)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    drive_type = DriveType.ADIABATIC if mode == "adiabatic" else DriveType.HEURISTIC
    cfg = make_quantum_config(size, drive_type)

    t0 = time.perf_counter()
    try:
        sol = run_solver(QUBOInstance(coefficients=Q), cfg)
        best, prob = best_cost_and_prob(sol)
        dt = time.perf_counter() - t0
        return {
            "size": size,
            "idx": idx,
            "mode": mode,
            "best_cost": best,
            "prob_best": prob,
            "time_s": dt,
            "status": "OK",
        }
    except Exception as e:
        dt = time.perf_counter() - t0
        return {
            "size": size,
            "idx": idx,
            "mode": mode,
            "best_cost": None,
            "prob_best": None,
            "time_s": dt,
            "status": f"FAIL: {e}",
        }


# ============================================================
# MAIN
# ============================================================

def main():
    datasets = load_datasets_by_size(DATASET_DIR)

    # Phase 1: CPLEX baseline
    cplex = compute_cplex_baseline(datasets)

    # Build quantum tasks (adiabatic + heuristic)
    tasks: List[Tuple[int, int, torch.Tensor, str]] = []
    for size, ds in datasets.items():
        for idx in range(len(ds)):
            Q, _ = ds[idx]
            Q = Q.to(dtype=torch.float32, device="cpu")
            tasks.append((size, idx, Q, "adiabatic"))
            tasks.append((size, idx, Q, "heuristic"))

    # CSV init
    fieldnames = [
        "qubo_size",
        "qubo_index",
        "best_cost_adiabatic",
        "best_cost_heuristic",
        "best_cost_cplex",
        "prob_best_adiabatic",
        "prob_best_heuristic",
    ]

    ctx = get_context(MP_CONTEXT)
    max_workers = min(MAX_WORKERS_CAP, os.cpu_count() or 1, len(tasks))
    max_workers = max(1, max_workers)
    print(f"\n=== Phase 2: Quantum solves (parallel) | max_workers={max_workers} ===")

    # Accumulate results per (size, idx)
    acc: Dict[Tuple[int, int], Dict[str, Any]] = {}

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futs = [ex.submit(solve_quantum_task, t) for t in tasks]

            for fut in as_completed(futs):
                res = fut.result()
                key = (res["size"], res["idx"])
                slot = acc.setdefault(key, {})
                slot[res["mode"]] = res

                # dès qu’on a les deux modes, on écrit une ligne
                if "adiabatic" in slot and "heuristic" in slot:
                    a = slot["adiabatic"]
                    h = slot["heuristic"]

                    writer.writerow({
                        "qubo_size": key[0],
                        "qubo_index": key[1],
                        "best_cost_adiabatic": a["best_cost"],
                        "best_cost_heuristic": h["best_cost"],
                        "best_cost_cplex": cplex.get(key, float("nan")),
                        "prob_best_adiabatic": a["prob_best"],
                        "prob_best_heuristic": h["prob_best"],
                    })
                    f.flush()
                    os.fsync(f.fileno())

                    print(
                        f"[DONE] size={key[0]} idx={key[1]} | "
                        f"AD={a['best_cost']} (p={a['prob_best']}) | "
                        f"HE={h['best_cost']} (p={h['prob_best']}) | "
                        f"CPLEX={cplex.get(key)}"
                    )
                    del acc[key]

    print(f"\n✅ CSV written to: {CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()
