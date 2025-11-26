from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import networkx as nx
import numpy as np
from numpy import format_float_scientific
import pandas as pd

from ._helpers import normalized_interaction


def eformat(f: Any) -> str:
    if 1 <= abs(f) < 1000:
        return f"{np.round(f, decimals=1)}"
    elif 0.01 <= abs(f):
        return f"{np.round(f, decimals=2)}"
    if f == 0:
        return "0"

    return format_float_scientific(f, exp_digits=1, precision=0)  # type: ignore


def get_ax(ax: Axes | None) -> Axes:
    if ax is None:
        return plt.gca()
    return ax


def draw_weighted_graph(
    graph: nx.Graph,
    thresholds: list = [0, 0.3, 0.6],
    edge_labels: Optional[dict] = None,
    ax: Axes | None = None,
) -> None:
    plt.figure(1, figsize=(30, 12), dpi=60)
    ax = get_ax(ax)

    print(f"{thresholds=}")
    t0, t1, t2 = thresholds
    elarge = [(u, v) for (u, v, w) in graph.edges.data("weight") if t2 < w]  # type: ignore
    esmall = [
        (u, v) for (u, v, w) in graph.edges.data("weight") if t1 < w <= t2 and t0 < w  # type: ignore
    ]
    etiny = [(u, v) for (u, v, w) in graph.edges.data("weight") if t0 <= w <= t1]  # type: ignore

    pos_all_dims = dict(graph.nodes.data("pos"))  # type: ignore
    pos = {k: v[0:2] for k, v in pos_all_dims.items()}

    ax.set_aspect("equal", adjustable="box")

    # nodes
    nx.draw_networkx_nodes(graph, pos, node_size=700, ax=ax)

    # edges
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6, ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=esmall,
        width=6,
        alpha=0.5,
        edge_color="b",
        style="dashed",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=etiny,
        width=4,
        alpha=0.3,
        edge_color="g",
        style="dashed",
        ax=ax,
    )

    # node labels
    nx.draw_networkx_labels(graph, pos, font_size=20, font_family="sans-serif", ax=ax)

    for _, coords in pos_all_dims.items():
        plt.text(
            coords[0] + 0.1,
            coords[1] + 0.1,
            "(" + ", ".join(eformat(coord) for coord in coords) + ")",
            fontsize=15,
            color="black",
        )

    # edge weight labels
    edge_labels = {
        k: f'{eformat(v)} {edge_labels[k] if edge_labels else ""}'
        for k, v in nx.get_edge_attributes(graph, "weight").items()
        if v >= t0
    }
    pos = {k: [vi + 1 / 2 for vi in v] for k, v in pos.items()}

    print(f"{pos=}")

    nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax, font_size=15)

    ax.margins(0.08)
    ax.axis("off")


def draw_set_graph_coords(
    graph: nx.Graph, coords: np.ndarray, edge_labels: Optional[dict] = None
) -> None:
    """coords are positions in numerical order of the nodes"""
    nx.set_node_attributes(graph, dict(enumerate(coords)), "pos")
    draw_weighted_graph(graph, edge_labels=edge_labels)
    plt.show()


def draw_graph_including_actual_weights(qubo_graph: nx.Graph, positions: np.ndarray) -> None:
    from IPython.display import display

    new_weights_matrix = np.full((len(qubo_graph), len(qubo_graph)), fill_value="", dtype=object)
    new_weights = dict()
    for u, v in qubo_graph.edges:
        dist = np.linalg.norm(positions[u] - positions[v])
        new_weights[(u, v)] = normalized_interaction(dist=float(dist))
        new_weights_matrix[min(u, v), max(u, v)] = eformat(new_weights[(u, v)])

    df = pd.DataFrame(new_weights_matrix)

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
    ):
        display(df)

    draw_set_graph_coords(
        graph=qubo_graph,
        coords=positions,
    )
