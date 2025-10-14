# tests/test_greedy_annimation.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from qoolqit._solvers.types import DeviceType

from qubosolver.algorithms.greedy.greedy import Greedy
from qubosolver.qubo_types import LayoutType


def _toy_qubo() -> torch.Tensor:
    # Petit QUBO dense et déterministe (5x5)
    return torch.tensor(
        [
            [-4.0, 1.0, 2.0, 1.5, 2.3],
            [1.0, -3.2, 1.7, 2.1, 0.8],
            [2.0, 1.7, -5.1, 0.7, 2.5],
            [1.5, 2.1, 0.7, -2.7, 1.2],
            [2.3, 0.8, 2.5, 1.2, -6.3],
        ],
        dtype=torch.float32,
    )


def _base_params(n: int) -> Dict[str, Any]:
    # Paramètres de base, sans animation par défaut
    return {
        "layout": LayoutType.TRIANGULAR,
        "traps": n + 4,
        "spacing": 5.0,
        "device": DeviceType.DIGITAL_ANALOG_DEVICE.value,
    }


def test_greedy_coords_shape_no_animation() -> None:
    Q = _toy_qubo()
    n = Q.shape[0]
    params = _base_params(n)
    params["draw_steps"] = False
    params["animation"] = False

    _, _, coords, _, _ = Greedy().launch_greedy(Q=Q, params=params)

    assert isinstance(coords, torch.Tensor)
    assert tuple(coords.shape) == (n, 2)


def test_greedy_animation_calls_renderer(monkeypatch: Any) -> None:
    """
    Force le chemin d’animation en patchant:
      - le flag _VIZ_OK à True (sinon pas d’appel)
      - la méthode _render_animation pour compter les appels
    Aucun fichier n’est écrit, aucun backend graphique requis.
    """
    Q = _toy_qubo()
    n = Q.shape[0]
    params = _base_params(n)
    params["draw_steps"] = True
    params["animation"] = True
    params["animation_save_path"] = None  # ne rien écrire

    # Compteur d’appels
    called: Dict[str, int] = {"count": 0}

    def _fake_render(
        self: Greedy,
        frames: List[Dict[str, Any]],
        all_coords_np: Any,
        spacing: float,
        layout_name: str,
        top_k: int = 5,
        save_path: Optional[str] = None,
        fps: float = 1.25,
    ) -> None:
        called["count"] += 1
        return None

    # 1) S'assurer que le code rentre bien dans le bloc d'animation
    import qubosolver.algorithms.greedy.greedy as greedy_mod

    monkeypatch.setattr(greedy_mod, "_VIZ_OK", True, raising=False)

    # 2) Remplacer le renderer par un stub inoffensif
    monkeypatch.setattr(Greedy, "_render_animation", _fake_render, raising=True)

    Greedy().launch_greedy(Q=Q, params=params)

    assert called["count"] >= 1
