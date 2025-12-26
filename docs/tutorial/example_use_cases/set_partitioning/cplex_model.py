from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import cplex
import numpy as np
from helpers import (
    bitstring_from_vals,
    chosen_list,
    coverage_report,
    feasibility_violation,
)


def create_cplex_model(
    x_names: list[str], nb_partitions: int, w: np.ndarray, nb_items: int, A: np.ndarray
) -> cplex.Cplex:
    mdl = cplex.Cplex()
    # Silence logs
    mdl.set_log_stream(None)
    mdl.set_error_stream(None)
    mdl.set_warning_stream(None)
    mdl.set_results_stream(None)

    mdl.set_problem_type(cplex.Cplex.problem_type.MILP)
    mdl.objective.set_sense(mdl.objective.sense.minimize)

    mdl.variables.add(obj=w.tolist(), types=["B"] * nb_partitions, names=x_names)

    # Constraints: exact cover for each item i
    for i in range(nb_items):
        idx = [x_names[p] for p in range(nb_partitions) if A[i, p] == 1]
        val = [1.0] * len(idx)
        mdl.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=idx, val=val)], senses=["E"], rhs=[1.0]
        )

    return mdl


def objective_cost(vals: list[float], w: np.ndarray) -> float:
    return float(np.dot(w, np.array(vals)))


def pprint_solution(
    vals: list[float],
    subsets: list[str],
    items: list[str],
    nb_items: int,
    nb_partitions: int,
    A: np.ndarary,
    w: np.ndarray,
    title: str = "Solution",
) -> None:
    vals = list(vals)
    print(f"--- {title} ---")
    print("Variables:", ", ".join(subsets))
    print("Optimal Bitstring        :", bitstring_from_vals(vals))
    print("Selected subsets :", chosen_list(vals, subsets))
    print("ILP cost         :", objective_cost(vals, w))
    print("Feasibility vio. :", feasibility_violation(vals, A))
    print("Coverage:")
    print(coverage_report(vals, subsets, items, nb_items, nb_partitions, A))


def solve_cplex(
    mdl: cplex.Cplex,
    x_names: list[str],
    subsets: list[str],
    items: list[str],
    nb_items: int,
    nb_partitions: int,
    A: np.ndarray,
    w: np.ndarray,
) -> Any:
    t0 = time.perf_counter()
    mdl.solve()
    solve_time = time.perf_counter() - t0

    status_str = mdl.solution.get_status_string()
    obj_val = mdl.solution.get_objective_value()
    x_star = mdl.solution.get_values(x_names)

    print(f"=== OPTIMAL SOLUTION (SPP, {nb_items} vars) ===")
    print("Status   :", status_str)
    print("Objective:", obj_val)
    best_ilp_cost = obj_val
    pprint_solution(
        x_star, subsets, items, nb_items, nb_partitions, A, w, title="Optimal"
    )

    print(f"\nExecution time: {solve_time:.4f} s")
    return best_ilp_cost


def populate_solution_pool(
    mdl: cplex.Cplex,
    x_names: list[str],
    subsets: list[str],
    A: np.ndarray,
) -> None:
    try:
        mdl.parameters.mip.pool.intensity.set(4)
        mdl.parameters.mip.pool.replace.set(2)
        mdl.parameters.mip.pool.capacity.set(100)
        mdl.parameters.mip.pool.relgap.set(1e75)
        mdl.parameters.mip.pool.absgap.set(1e75)
        mdl.populate_solution_pool()
    except cplex.exceptions.CplexError:
        pass

    num_solutions = mdl.solution.pool.get_num()

    # Collect unique solutions by bitstring
    unique = OrderedDict()
    for k in range(num_solutions):
        vals_k = mdl.solution.pool.get_values(k, x_names)
        bitstr = bitstring_from_vals(vals_k)
        cost_k = float(mdl.solution.pool.get_objective_value(k))
        if bitstr not in unique:
            unique[bitstr] = (cost_k, vals_k)

    # Sort by objective
    sorted_solutions = sorted(unique.items(), key=lambda kv: kv[1][0])

    print(
        f"\n=== SOLUTION POOL (unique = {len(sorted_solutions)}, \
        total CPLEX pool = {num_solutions}) ==="
    )
    max_show = min(10, len(sorted_solutions))
    for rank, (bitstr, (cost_k, vals_k)) in enumerate(
        sorted_solutions[:max_show], start=1
    ):
        print(f"\n-- Solution #{rank} --")
        print("Objective:", cost_k)
        print("Bitstring:", bitstr, "(order:", ",".join(subsets) + ")")
        print("Selected :", chosen_list(vals_k, subsets))
        print("Feas.Vio.:", feasibility_violation(vals_k, A))
