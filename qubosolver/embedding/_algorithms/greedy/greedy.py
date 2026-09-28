"""Greedy embedding of a QUBO matrix onto a fixed lattice of traps."""

from __future__ import annotations

import contextlib
import typing
from collections.abc import Callable
from typing import Any, cast

import torch

from qubosolver import Matrix, Tensor, Vector, Vectori, matrix, tensor, vector, vectori

from .layout import get_layout

# Optional imports for animation; guarded so library usage stays safe in non-notebook envs.
try:  # pragma: no cover
    import numpy as np

    _VIZ_OK = True
except Exception:  # pragma: no cover
    _VIZ_OK = False


@typing.no_type_check
class Greedy:
    """Greedy embedding on a fixed lattice (triangular or square).

    At each step, place one logical node onto one trap to minimize the
    incremental mismatch between the logical QUBO matrix Q and the physical
    interaction matrix U (approx. 1 / ||r_i - r_j||^6).

    Adds:
      - optional `on_step(state: dict)` callback for instrumentation
      - post-run animation when params["animation"] or params["draw_steps"] is True
    """

    # ----------------------------
    # Layout utilities
    # ----------------------------
    def get_predefined_coordinates(self, params: dict) -> Tensor:
        """Build the initial lattice of trap coordinates.

        Expected `params` keys:
          - "layout": Layout (TRIANGULAR or SQUARE) or "triangular"/"square"
          - "traps": int (number of trap sites)
          - "spacing": float (minimum inter-site spacing)
        """
        type_layout = params["layout"]
        n_traps: int = params["traps"]
        spacing: float = params["spacing"]

        return spacing * get_layout(layout_type=type_layout, n_traps=n_traps)

    # ----------------------------
    # Interaction matrix
    # ----------------------------
    def interaction_matrix(self, coordinates: Tensor) -> Matrix:
        """Interaction between traps, U[p, q] = 1 / ||r_p - r_q|| ** 6.

        The diagonal is left at zero: a trap holds at most one node, so a node
        is never compared against itself.
        """
        n_traps = len(coordinates)
        U = matrix.zeros(n_traps)
        if n_traps < 2:
            return U

        distances = (coordinates[:, None, :] - coordinates[None, :, :]).norm(dim=-1)
        off_diagonal = ~torch.eye(n_traps, dtype=torch.bool, device=U.device)
        U[off_diagonal] = 1.0 / distances[off_diagonal] ** 6
        return U

    # ----------------------------
    # Next node heuristic
    # ----------------------------
    def get_best(self, couplings: Vector, placed: Tensor) -> int:
        """Pick the next logical node: the unplaced one most coupled to the placed set.

        Args:
            couplings: `couplings[i] = sum of Q[i, j] over the placed nodes j`,
                maintained incrementally by the caller.
            placed: Boolean mask of the already-placed nodes.

        Returns:
            The index of the chosen node. Ties go to the lowest index.
        """
        return int(torch.argmax(couplings.masked_fill(placed, -torch.inf)))

    # ----------------------------
    # Best trap for a node
    # ----------------------------
    def optimize_position(
        self,
        U: Matrix,
        Q: Matrix,
        u: int,
        placed_nodes: Vectori,
        placed_traps: Vectori,
        available_traps: Vectori,
        return_candidates: bool = False,
    ) -> tuple[int, float, list[tuple[int, float]]]:
        """Evaluate every free trap p for node u and pick the one that minimizes s(p).

            s(p) = sum_{j placed} | Q[u, j] - U[p, trap(j)] |.

        Args:
            U: Trap-trap interaction matrix.
            Q: Logical QUBO matrix.
            u: Node to place.
            placed_nodes: Indices of the already-placed nodes.
            placed_traps: `placed_traps[k]` is the trap holding `placed_nodes[k]`.
            available_traps: Indices of the free traps, in ascending order.
            return_candidates: Also report the score of every free trap.

        Returns:
            `(choice_p, min_val, candidates)`, where `candidates` is
            `[(trap_index, incremental_mismatch), ...]` when *return_candidates*
            is set and empty otherwise. Ties go to the lowest trap index.
        """
        # (n_available, n_placed) block of deviations, reduced over the placed nodes.
        u_couplings = Q[u].index_select(0, placed_nodes)
        trap_couplings = U.index_select(0, available_traps).index_select(1, placed_traps)
        scores = (u_couplings - trap_couplings).abs().sum(dim=1, dtype=torch.float64)

        best = int(torch.argmin(scores))
        choice_p = int(available_traps[best])
        min_val = float(scores[best])

        candidates = (
            list(zip(available_traps.tolist(), scores.tolist(), strict=True))
            if return_candidates
            else []
        )
        return choice_p, min_val, candidates

    @staticmethod
    def _emit_step(
        on_step: Callable[[dict[str, Any]], None] | None,
        **snapshot: Any,  # noqa: ANN401 (heterogeneous snapshot payload forwarded verbatim)
    ) -> None:  # pragma: no cover
        """Forward a state snapshot to `on_step`, never letting viz crash the solver."""
        if on_step is None:
            return
        with contextlib.suppress(Exception):
            on_step(snapshot)

    # ----------------------------
    # Main greedy pass for one start node
    # ----------------------------
    def greedy_algorithm(
        self,
        U: Matrix,
        Q: Matrix,
        coords: Tensor,
        v: int,
        results: dict,
        params: dict,
        on_step: Callable[[dict[str, Any]], None] | None = None,
        max_radial_distance: float = torch.inf,
    ) -> dict:
        """Greedy loop starting from node v.

        If `on_step` is provided, emit a state snapshot after each placement (and an initial
        snapshot).
        """
        n_nodes: int = Q.shape[0]
        n_traps: int = len(coords)
        n_extra_traps: int = max(n_traps - n_nodes, 0)

        # Placement state, in placement order: only the first `n_placed` entries
        # of `placed_nodes` / `placed_traps` are meaningful.
        placed_nodes: Vectori = vectori.zeros(n_nodes)
        placed_traps: Vectori = vectori.zeros(n_nodes)
        n_placed: int = 0
        placed_mask = torch.zeros(n_nodes, dtype=torch.bool, device=tensor.device())
        free_traps = torch.ones(n_traps, dtype=torch.bool, device=tensor.device())
        # couplings[i] = sum of Q[i, j] over the placed nodes j, kept incrementally
        couplings: Vector = vector.zeros(n_nodes)

        def _place(node: int, trap: int) -> None:
            nonlocal n_placed
            placed_nodes[n_placed] = node
            placed_traps[n_placed] = trap
            n_placed += 1
            placed_mask[node] = True
            free_traps[trap] = False
            couplings.add_(Q[:, node])

        step_id = 0
        total_mismatch = 0.0

        def _snapshot(  # pragma: no cover
            node: int, trap: int, inc_mismatch: float, candidates: list[tuple[int, float]]
        ) -> dict[str, Any]:
            """Build the instrumentation payload for the current state."""
            nodes_so_far = placed_nodes[:n_placed].tolist()
            traps_so_far = placed_traps[:n_placed].tolist()
            return {
                "step": step_id,
                "picked_node": int(node),
                "picked_trap": int(trap),
                "placed_nodes": nodes_so_far,
                "used_traps": torch.nonzero(~free_traps).squeeze(1).tolist(),
                "inc_mismatch": float(inc_mismatch),
                "total_mismatch": float(total_mismatch),
                "per_trap_candidates": candidates,
                "positioned_coords": {
                    node_id: tuple(coords[trap_id].tolist())
                    for node_id, trap_id in zip(nodes_so_far, traps_so_far, strict=True)
                },
                "trap_of": dict(zip(nodes_so_far, traps_so_far, strict=True)),
            }

        # the start node goes to the trap closest to the origin
        origin_trap = int(torch.argmin(coords.square().sum(dim=1)))
        _place(v, origin_trap)
        if on_step is not None:  # pragma: no cover
            self._emit_step(on_step, **_snapshot(v, origin_trap, 0.0, []))

        # If visualization is enabled, ask for the per-trap scores too
        want_candidates = bool(params.get("draw_steps", False) or (on_step is not None))

        while n_placed < n_nodes:
            u = self.get_best(couplings, placed_mask)
            available_traps = torch.nonzero(free_traps).squeeze(1)
            p, inc_val, candidates = self.optimize_position(
                U,
                Q,
                u,
                placed_nodes[:n_placed],
                placed_traps[:n_placed],
                available_traps,
                return_candidates=want_candidates,
            )
            candidates.sort(key=lambda t: t[1])  # ascending by mismatch

            # check whether trap coordinate is within the maximal radial distance
            if float(coords[p].norm()) >= max_radial_distance:
                if n_extra_traps == 0:
                    raise ValueError(
                        f"no traps found to place qubit '{u}' "
                        f"within {max_radial_distance} of origin."
                    )

                free_traps[p] = False
                n_extra_traps -= 1
                if on_step is not None:  # pragma: no cover
                    self._emit_step(on_step, **_snapshot(u, p, 0.0, candidates))
                continue

            # commit placement; the winning score is exactly the incremental mismatch
            _place(u, p)
            total_mismatch += inc_val
            step_id += 1
            if on_step is not None:  # pragma: no cover
                self._emit_step(on_step, **_snapshot(u, p, inc_val, candidates))

        # finalize coordinates tensor
        final_coords = tensor.zeros(n_nodes, 2)
        final_coords[placed_nodes] = coords[placed_traps]

        iu, ju = torch.triu_indices(n_nodes, n_nodes, offset=1)
        uij = 1 / torch.cdist(final_coords, final_coords)[iu, ju] ** 6
        diff = float(torch.abs(Q[iu, ju] - uij).sum(dtype=torch.float64))

        results[v] = {"coords": final_coords, "distance": diff}
        return results

    # ----------------------------
    # Internal: post-run animation (only if animation=True)
    # ----------------------------
    def _render_animation(  # pragma: no cover  # noqa: C901 (viz setup, not worth splitting)
        self,
        frames: list[dict[str, Any]],
        all_coords_np: np.ndarray,
        spacing: float,
        layout_name: str,
        top_k: int = 5,
        save_path: str | None = None,
        fps: float = 1.25,
    ) -> Any | None:  # noqa: ANN401 (matplotlib animation type is an optional, lazily-imported dep)
        """Post-run animation (traps = gray, qubits = green). No persistent rings."""
        if not _VIZ_OK:
            return None  # numpy not available

        import os

        import numpy as np

        try:
            import matplotlib.pyplot as plt  # deptry: ignore[DEP004]
            from IPython.display import HTML, display  # deptry: ignore[DEP004]
            from matplotlib import animation, gridspec  # deptry: ignore[DEP004]
            from matplotlib.animation import FFMpegWriter, PillowWriter  # deptry: ignore[DEP004]
        except ImportError as e:
            raise ImportError(
                "Rendering the greedy-embedding animation requires 'matplotlib' and "
                "'ipython'. Install them with: pip install 'qubo-solver[dev]'"
            ) from e

        X, Y = all_coords_np[:, 0], all_coords_np[:, 1]
        xmin, xmax = X.min() - spacing, X.max() + spacing
        ymin, ymax = Y.min() - spacing, Y.max() + spacing

        # ---- Figure & axes
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[5.2, 1.8], hspace=0.15)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[1, 0])

        # ---- Main canvas
        ax_top.set_aspect("equal", adjustable="box")
        ax_top.set_title("Greedy embedding algo demo", fontsize=13, pad=10)
        ax_top.set_xlim(xmin, xmax)
        ax_top.set_ylim(ymin, ymax)
        ax_top.grid(True, alpha=0.20)
        # Traps (subtle gray)
        ax_top.scatter(X, Y, s=28, color="#bdbdbd", alpha=0.45, zorder=1)

        # Placed qubits (green with black edge)
        placed_scatter = ax_top.scatter(
            [], [], s=130, color="tab:green", edgecolor="k", linewidths=0.8, zorder=3
        )
        # We manage labels manually to remove/recreate them each frame
        labels: list[Any] = []

        # ---- Info panel
        ax_info.axis("off")
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)

        label_x, value_x = 0.04, 0.34
        y0, dy = 0.86, 0.22

        ax_info.text(
            label_x,
            y0 - 0 * dy,
            "Step",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        ax_info.text(
            label_x,
            y0 - 1 * dy,
            "Last placement",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        ax_info.text(
            label_x,
            y0 - 2 * dy,
            "Mismatch",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        ax_info.text(
            label_x,
            y0 - 3 * dy,
            "Total mismatch",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        # Right column: Top-k
        ax_info.text(
            0.58,
            y0 - 0 * dy,
            f"Top-{top_k} candidates",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        val_step = ax_info.text(value_x, y0 - 0 * dy, "", ha="left", va="center", fontsize=11)
        val_last = ax_info.text(value_x, y0 - 1 * dy, "", ha="left", va="center", fontsize=11)
        val_inc = ax_info.text(value_x, y0 - 2 * dy, "", ha="left", va="center", fontsize=11)
        val_total = ax_info.text(value_x, y0 - 3 * dy, "", ha="left", va="center", fontsize=11)

        # Vertical list of candidates (avoid overlap)
        val_cand = ax_info.text(
            0.58, y0 - 0.95 * dy, "", ha="left", va="top", fontsize=11, linespacing=1.35
        )

        def init() -> tuple[Any, Any, Any, Any, Any, Any]:
            placed_scatter.set_offsets(np.empty((0, 2)))
            for t in labels:
                t.remove()
            labels.clear()
            val_step.set_text("")
            val_last.set_text("")
            val_inc.set_text("")
            val_total.set_text("")
            val_cand.set_text("")
            return (placed_scatter, val_step, val_last, val_inc, val_total, val_cand)

        def update(i: int) -> tuple[Any, ...]:
            st = frames[i]

            # Clear previous labels
            for t in labels:
                t.remove()
            labels.clear()

            # Update placed points + labels
            trap_of = st.get("trap_of", {})
            pos = []
            for _, trap_idx in trap_of.items():
                if trap_idx is None or trap_idx < 0 or trap_idx >= len(all_coords_np):
                    continue
                pos.append(all_coords_np[trap_idx])
            if pos:
                placed_scatter.set_offsets(np.array(pos))
                # (re)create labels above points
                for q_idx, trap_idx in trap_of.items():
                    if trap_idx is None or trap_idx < 0 or trap_idx >= len(all_coords_np):
                        continue
                    x, y = all_coords_np[trap_idx]
                    labels.append(
                        ax_top.text(
                            x,
                            y,
                            str(q_idx),
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="white",
                            zorder=4,
                        )
                    )
            else:
                placed_scatter.set_offsets(np.empty((0, 2)))

            # Update info panel
            val_step.set_text(f"{st.get('step', 0)}")
            val_last.set_text(
                f"qubit {st.get('picked_node', '-')} → trap {st.get('picked_trap', '-')}"
            )
            val_inc.set_text(f"{st.get('inc_mismatch', 0.0):.4f}")
            val_total.set_text(f"{st.get('total_mismatch', 0.0):.4f}")

            # Vertical list of top candidates
            top = st.get("per_trap_candidates", [])[:top_k]
            bullets = "\n".join([f"• trap {p}  ({inc:.3f})" for p, inc in top]) if top else "—"
            val_cand.set_text(bullets)

            return (
                placed_scatter,
                val_step,
                val_last,
                val_inc,
                val_total,
                val_cand,
                *labels,
            )

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(frames),
            init_func=init,
            interval=4000,  # derive interval from fps; prevents 0 division
            blit=False,
            repeat=False,
        )

        # ---- Save to disk if requested
        if save_path is not None:
            try:
                # Ensure directory exists
                folder = os.path.dirname(save_path)
                if folder and not os.path.exists(folder):
                    os.makedirs(folder, exist_ok=True)

                ext = os.path.splitext(save_path)[1].lower()

                if ext in (".mp4", ""):
                    # Prefer explicit FFMpegWriter for clearer errors.
                    if animation.writers.is_available("ffmpeg"):
                        ffmpeg_writer = FFMpegWriter(
                            fps=cast(int, fps), bitrate=1800, metadata={"artist": "qubo-solver"}
                        )
                        target = save_path if ext == ".mp4" else save_path + ".mp4"
                        anim.save(target, writer=ffmpeg_writer, dpi=180)
                        print(f"[anim] MP4 saved to: {target}")
                    else:
                        raise RuntimeError(
                            "ffmpeg is not available in PATH. Install ffmpeg or export GIF instead."
                        )
                elif ext == ".gif":
                    # PillowWriter avoids requiring ImageMagick.
                    pillow_writer = PillowWriter(fps=cast(int, fps))
                    anim.save(save_path, writer=pillow_writer, dpi=180)
                    print(f"[anim] GIF saved to: {save_path}")
                else:
                    # Unknown extension -> default to MP4
                    if animation.writers.is_available("ffmpeg"):
                        ffmpeg_writer = FFMpegWriter(
                            fps=cast(int, fps), bitrate=1800, metadata={"artist": "qubo-solver"}
                        )
                        target = save_path + ".mp4"
                        anim.save(target, writer=ffmpeg_writer, dpi=180)
                        print(f"[anim] MP4 (default) saved to: {target}")
                    else:
                        raise RuntimeError(
                            f"Unsupported extension '{ext}' and ffmpeg not available."
                        )
            except Exception as e:
                # Do not swallow errors; print a helpful message instead
                print(f"[anim] Save failed: {e}")

        # ---- Inline HTML preview (safe to fail)
        try:
            display(HTML(anim.to_jshtml()))
        except Exception as e:
            print(f"[anim] to_jshtml failed: {e}")

        # Close the figure ONLY after saving and HTML rendering
        plt.close(fig)

        return anim

    # ----------------------------
    # Entry point for the pipeline
    # ----------------------------
    def launch_greedy(
        self,
        Q: torch.Tensor,
        *,
        max_min_dist_ratio: float,
        params: dict,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[Any, torch.Tensor]:
        """Run greedy from each start node and keep the best result.

        Instrumentation rules:
          - If params['animation'] or params['draw_steps'] is True, we collect steps and
            render a post-run animation automatically.
          - If `on_step` is provided, we still instrument but do not necessarily render.
          - Else, no instrumentation (zero overhead).

        Returns:
          (best_result_item, coords)
        """
        n_traps = params["traps"]
        n_nodes = Q.shape[0]

        if n_traps < n_nodes:
            raise ValueError(f"Not enough traps ({n_traps}) to position {n_nodes} nodes.")

        coordinates = self.get_predefined_coordinates(params)
        U = self.interaction_matrix(coordinates)
        max_radial_distance = max_min_dist_ratio * float(params["spacing"])

        results: dict = {}

        # Decide instrumentation/animation from params only
        anim_flag = bool(params.get("animation", False) or params.get("draw_steps", False))
        instrument = bool(params.get("draw_steps", False) or on_step is not None or anim_flag)

        frames: list[dict[str, Any]] = []

        if instrument:  # pragma: no cover

            def _collector(state: dict[str, Any]) -> None:
                if on_step is not None:
                    with contextlib.suppress(Exception):  # never let viz crash
                        on_step(state)
                if anim_flag:
                    try:
                        frames.append(state.copy())
                    except Exception:
                        frames.append(state)

            cb = _collector
        else:
            cb = None

        for node in range(n_nodes):
            self.greedy_algorithm(
                U,
                Q,
                coords=coordinates,
                v=node,
                results=results,
                params=params,
                on_step=cb,
                max_radial_distance=max_radial_distance,
            )

        best_result = min(results.items(), key=lambda x: x[1]["distance"])
        coords = best_result[1]["coords"]

        # Post-run animation if requested
        if anim_flag and frames and _VIZ_OK:  # pragma: no cover
            # Rebuild full lattice coords to show ALL traps (including extras)
            all_coords_np = (
                coordinates.numpy() if hasattr(coordinates, "numpy") else np.array(coordinates)
            )
            self._render_animation(
                frames=frames,
                all_coords_np=all_coords_np,
                spacing=float(params["spacing"]),
                layout_name=str(params["layout"]),
                top_k=int(params.get("animation_top_k", 5)),
                save_path=params.get("animation_save_path"),
                fps=0.5,
            )

        return best_result, coords
