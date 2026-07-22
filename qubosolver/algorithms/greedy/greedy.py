from __future__ import annotations

import copy
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from pulser.register.register_layout import RegisterLayout

from qubosolver.algorithms.greedy.layout import get_layout

# Optional imports for animation; guarded so library usage stays safe in non-notebook envs.
try:  # pragma: no cover
    import numpy as np

    _VIZ_OK = True
except Exception:  # pragma: no cover
    _VIZ_OK = False


@typing.no_type_check
class Greedy:
    """
    Greedy embedding on a fixed lattice (triangular or square).

    At each step, place one logical node onto one trap to minimize the
    incremental mismatch between the logical QUBO matrix Q and the physical
    interaction matrix U (approx. C / ||r_i - r_j||^6).

    Adds:
      - optional `on_step(state: dict)` callback for instrumentation
      - post-run animation when params["animation"] or params["draw_steps"] is True
    """

    MAPPING_COORDS_POSITIONS: dict = {}
    MAPPING_POSITIONS_COORDS: dict = {}

    # ----------------------------
    # Layout utilities
    # ----------------------------
    def get_predefined_coordinates(self, params: dict) -> tuple[RegisterLayout, torch.Tensor]:
        """
        Build the initial Pulser layout and return its coordinates.

        Expected `params` keys:
          - "layout": LayoutType (TRIANGULAR or SQUARE) or "triangular"/"square"
          - "traps": int (number of trap sites)
          - "spacing": float (minimum inter-site spacing)
        """
        type_layout = params["layout"]
        n_traps: int = params["traps"]
        spacing: float = params["spacing"]

        coords = spacing * get_layout(layout_type=type_layout, n_traps=n_traps)

        # build fast maps coord <-> trap index
        self.MAPPING_COORDS_POSITIONS.clear()
        self.MAPPING_POSITIONS_COORDS.clear()
        for i, coord in enumerate(coords.tolist()):
            self.MAPPING_COORDS_POSITIONS[tuple(coord)] = i
            self.MAPPING_POSITIONS_COORDS[i] = coord

        return coords

    # ----------------------------
    # Precompute mismatch tensor
    # ----------------------------
    def precompute_coefficients(self, Q: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Precompute Z[i,j,p,q] = | Q[i,j] - U[p,q] | where U[p,q] is the
        physical interaction between traps p and q (C / r^6).
        """
        n_nodes = Q.shape[0]
        n_traps = len(coordinates)

        # Physical interaction matrix U on traps
        U = torch.zeros((n_traps, n_traps), dtype=torch.float32)
        for p in range(n_traps):
            for q in range(p + 1, n_traps):
                U[p, q] = 1 / torch.norm(coordinates[p] - coordinates[q]) ** 6
                U[q, p] = U[p, q]

        # Z: node-node vs trap-trap mismatch
        Z = torch.zeros((n_nodes, n_nodes, n_traps, n_traps), dtype=torch.float32)
        p_idx, q_idx = torch.triu_indices(n_traps, n_traps, offset=1)
        # broadcast Q to all trap pairs, compare to U[p,q]
        diffs = torch.abs(Q[:, :, None].clone().detach() - U[p_idx, q_idx])
        Z[:, :, p_idx, q_idx] = diffs
        Z[:, :, q_idx, p_idx] = diffs

        return Z

    # ----------------------------
    # Next node heuristic
    # ----------------------------
    def get_best(self, Q: torch.Tensor, positioned: set, all_vertices: set) -> Any:
        """
        Pick the next logical node: the one with the largest total coupling
        to the already-positioned set.
        """
        all_vertices = all_vertices.difference(positioned)
        node_contributes: List[Tuple[int, float]] = []
        for u in all_vertices:
            s: float = 0.0
            for j in positioned:
                s += float(Q[u, j].item())
            node_contributes.append((u, s))
        u = list(sorted(node_contributes, key=lambda x: x[1], reverse=True))[0][0]
        return u

    # ----------------------------
    # Best trap for a node
    # ----------------------------
    def optimize_position(
        self,
        Z: torch.Tensor,
        u: int,
        positioned: set,
        positioned_coords: dict,
        all_traps: set,
        used_traps: set,
        return_candidates: bool = False,
    ) -> tuple[Any, Any, Any] | tuple[Any, Any, Any, List[Tuple[int, float]]]:
        """
        Evaluate all available traps p for node u and pick the one that minimizes:
            s(p) = sum_{j in positioned} Z[u, j, p, trap(j)].

        Returns (choice_p, choice_coordinates, min_val)
        or, if return_candidates=True:
                (choice_p, choice_coordinates, min_val, candidates)
            where candidates = [(trap_index, incremental_mismatch), ...]
        """
        available_traps = all_traps.difference(used_traps)

        i = u
        choice_p: int = -1
        choice_coordinates: tuple = (None, None)
        min_val: float = float("inf")
        candidates: List[Tuple[int, float]] = []

        for p in available_traps:
            s = 0.0
            for j in positioned:
                q = self.MAPPING_COORDS_POSITIONS[positioned_coords[j]]
                s += Z[i, j, p, q].item()

            candidates.append((p, float(s)))

            if s < min_val:
                min_val = float(s)
                choice_coordinates = tuple(self.MAPPING_POSITIONS_COORDS[p])
                choice_p = p

        if return_candidates:  # pragma: no cover
            return choice_p, choice_coordinates, min_val, candidates
        return choice_p, choice_coordinates, min_val

    # ----------------------------
    # Main greedy pass for one start node
    # ----------------------------
    def greedy_algorithm(
        self,
        Z: torch.Tensor,
        Q: torch.Tensor,
        coords: torch.Tensor,
        v: int,
        results: dict,
        params: dict,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_radial_distance: float = torch.inf,
    ) -> dict:
        """
        Greedy loop starting from node v. If `on_step` is provided, emit a
        state snapshot after each placement (and an initial snapshot).
        """
        nodes = list(range(Q.shape[0]))

        vertices = set(nodes)
        n_traps = len(coords)
        all_traps = set(range(n_traps))

        n: int = len(Q)
        n_extra_traps: int = 0
        init_coord: tuple = (0, 0)
        positioned: set = set([v])
        positioned_coords: dict = {v: init_coord}
        used_coords: set = set([init_coord])
        used_traps: set = set([self.MAPPING_COORDS_POSITIONS[init_coord]])

        if n_traps > n:
            n_extra_traps = n_traps - n

        # helpers for instrumentation
        def _trap_of_from_coords() -> Dict[int, int]:  # pragma: no cover
            out: Dict[int, int] = {}
            for node_id, coord in positioned_coords.items():
                out[node_id] = int(self.MAPPING_COORDS_POSITIONS[coord])
            return out

        step_id = 0
        total_mismatch = 0.0

        # initial snapshot (optional)
        if on_step is not None:  # pragma: no cover
            try:
                on_step(
                    {
                        "step": step_id,
                        "picked_node": int(v),
                        "picked_trap": int(self.MAPPING_COORDS_POSITIONS.get(init_coord, -1)),
                        "placed_nodes": list(positioned),
                        "used_traps": list(used_traps),
                        "inc_mismatch": 0.0,
                        "total_mismatch": 0.0,
                        "per_trap_candidates": [],
                        "positioned_coords": positioned_coords.copy(),
                        "trap_of": _trap_of_from_coords(),
                    }
                )
            except Exception:
                pass  # never let viz crash the solver

        while len(positioned) < len(nodes):

            u = self.get_best(Q, positioned, copy.deepcopy(vertices))

            # If visualization is enabled, ask for candidates too
            want_candidates = bool(params.get("draw_steps", False) or (on_step is not None))
            if want_candidates:  # pragma: no cover
                res4 = self.optimize_position(
                    Z=Z,
                    u=u,
                    positioned=positioned,
                    positioned_coords=positioned_coords,
                    all_traps=copy.deepcopy(all_traps),
                    used_traps=used_traps,
                    return_candidates=True,
                )
                # Help mypy: explicitly cast 4-tuple
                _, u_coordinates, _, candidates = typing.cast(
                    Tuple[Any, Any, Any, List[Tuple[int, float]]], res4
                )
                candidates.sort(key=lambda t: t[1])  # ascending by mismatch
            else:
                res3 = self.optimize_position(
                    Z=Z,
                    u=u,
                    positioned=positioned,
                    positioned_coords=positioned_coords,
                    all_traps=copy.deepcopy(all_traps),
                    used_traps=used_traps,
                    return_candidates=False,
                )
                # Help mypy: explicitly cast 3-tuple
                _, u_coordinates, _ = typing.cast(Tuple[Any, Any, Any], res3)
                candidates = []

            distance = torch.tensor(u_coordinates).norm().item()

            # check whether trap coordinate is within the maximal radial distance
            if distance >= max_radial_distance:
                if n_extra_traps == 0:
                    raise ValueError(
                        f"no traps found to place qubit '{u}' "
                        f"within {max_radial_distance}µm from origin."
                    )

                used_coords.add(u_coordinates)
                used_traps.add(self.MAPPING_COORDS_POSITIONS[u_coordinates])
                n_extra_traps -= 1
                # snapshot of the skip (optional)
                if on_step is not None:  # pragma: no cover
                    try:
                        on_step(
                            {
                                "step": step_id,
                                "picked_node": int(u),
                                "picked_trap": int(self.MAPPING_COORDS_POSITIONS[u_coordinates]),
                                "placed_nodes": list(positioned),
                                "used_traps": list(used_traps),
                                "inc_mismatch": 0.0,
                                "total_mismatch": float(total_mismatch),
                                "per_trap_candidates": candidates,
                                "positioned_coords": positioned_coords.copy(),
                                "trap_of": _trap_of_from_coords(),
                            }
                        )
                    except Exception:
                        pass
                continue

            # commit placement
            positioned_coords[u] = u_coordinates
            positioned.add(u)
            used_coords.add(u_coordinates)
            used_traps.add(self.MAPPING_COORDS_POSITIONS[u_coordinates])

            # incremental mismatch (recompute from Z for clarity)
            inc_val = 0.0
            for j in positioned:
                if j == u:
                    continue
                q = self.MAPPING_COORDS_POSITIONS[positioned_coords[j]]
                p = self.MAPPING_COORDS_POSITIONS[u_coordinates]
                inc_val += float(Z[u, j, p, q])

            total_mismatch += float(inc_val)
            step_id += 1

            # emit snapshot
            if on_step is not None:  # pragma: no cover
                try:
                    on_step(
                        {
                            "step": step_id,
                            "picked_node": int(u),
                            "picked_trap": int(self.MAPPING_COORDS_POSITIONS[u_coordinates]),
                            "placed_nodes": list(positioned),
                            "used_traps": list(used_traps),
                            "inc_mismatch": float(inc_val),
                            "total_mismatch": float(total_mismatch),
                            "per_trap_candidates": candidates,
                            "positioned_coords": positioned_coords.copy(),
                            "trap_of": _trap_of_from_coords(),
                        }
                    )
                except Exception:
                    pass

        # finalize coordinates tensor
        final_coords = torch.zeros((Q.shape[0], 2), dtype=torch.float32)
        for v2, coord in positioned_coords.items():
            final_coords[v2, 0] = coord[0]
            final_coords[v2, 1] = coord[1]

        positioned_coords.clear()
        positioned.clear()
        used_coords.clear()
        used_traps.clear()

        # compute final total distance (as in original code)
        diff = 0.0
        for i in range(Q.shape[0]):
            for j in range(i + 1, Q.shape[0]):
                uij = 1 / torch.norm(final_coords[i] - final_coords[j]) ** 6
                diff += abs(Q[i, j] - uij)

        results[v] = {"coords": final_coords, "distance": diff}
        return results

    # ----------------------------
    # Internal: post-run animation (only if animation=True)
    # ----------------------------
    def _render_animation(  # pragma: no cover
        self,
        frames: List[Dict[str, Any]],
        all_coords_np: "np.ndarray",
        spacing: float,
        layout_name: str,
        top_k: int = 5,
        save_path: Optional[str] = None,
        fps: float = 1.25,
    ) -> Optional[Any]:
        """Post-run animation (traps = gray, qubits = green). No persistent rings."""
        if not _VIZ_OK:
            return None  # matplotlib or numpy not available

        import os

        import matplotlib.pyplot as plt
        import numpy as np
        from IPython.display import HTML, display
        from matplotlib import animation, gridspec
        from matplotlib.animation import FFMpegWriter, PillowWriter

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
        labels: List[Any] = []

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
            for q_idx, trap_idx in trap_of.items():
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
                f"qubit {st.get('picked_node', '–')} → trap {st.get('picked_trap', '–')}"
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
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Any:
        """
        Run greedy from each start node and keep the best result.

        Instrumentation rules:
          - If params['animation'] or params['draw_steps'] is True, we collect steps and
            render a post-run animation automatically.
          - If `on_step` is provided, we still instrument but do not necessarily render.
          - Else, no instrumentation (zero overhead).

        Returns:
          (best_result_item, None, coords, r_cut, omega)
        """
        n_traps = params["traps"]
        n_nodes = Q.shape[0]

        if n_traps < n_nodes:
            raise ValueError(f"Not enough traps ({n_traps}) to position {n_nodes} nodes.")

        coordinates = self.get_predefined_coordinates(params)
        predefined_coordinates = coordinates.clone().detach()

        Z = self.precompute_coefficients(Q, predefined_coordinates)
        nodes = list(range(n_nodes))

        results: dict = {}

        # Decide instrumentation/animation from params only
        anim_flag = bool(params.get("animation", False) or params.get("draw_steps", False))
        instrument = bool(params.get("draw_steps", False) or on_step is not None or anim_flag)

        frames: List[Dict[str, Any]] = []

        if instrument:  # pragma: no cover

            def _collector(state: Dict[str, Any]) -> None:
                if on_step is not None:
                    try:
                        on_step(state)
                    except Exception:
                        pass
                if anim_flag:
                    try:
                        frames.append(state.copy())
                    except Exception:
                        frames.append(state)

            cb = _collector
        else:
            cb = None

        max_radial_distance = max_min_dist_ratio * float(params["spacing"])

        for node in nodes:
            self.greedy_algorithm(
                Z,
                Q,
                coords=predefined_coordinates,
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
            all_coords_t = self.get_predefined_coordinates(params)
            if hasattr(all_coords_t, "numpy"):
                all_coords_np = all_coords_t.numpy()
            else:
                all_coords_np = np.array(all_coords_t)
            self._render_animation(
                frames=frames,
                all_coords_np=all_coords_np,
                spacing=float(params["spacing"]),
                layout_name=str(params["layout"]),
                top_k=int(params.get("animation_top_k", 5)),
                save_path=params.get("animation_save_path", None),
                fps=0.5,
            )

        return best_result, coords
