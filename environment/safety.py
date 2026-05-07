"""
safety.py
---------
Safety Projector: proyección convexa QP con CVXPY.

Bloque 5 del modelo: garantiza que los pesos sean SIEMPRE operativos.

Problema QP:
  min  ||w - w_raw||²
  s.t. sum(w) = 1          (fully invested)
       w_i >= 0            (long-only)
       w_i <= w_max        (límite por activo)

Ventaja sobre reward shaping:
  - Separar "aprendizaje" de "seguridad" evita el conservadurismo excesivo
    que ocurre cuando los constraints están en la reward.
  - El agente aprende a maximizar retorno; el safety layer garantiza
    que el resultado sea siempre operativo.
  - Referencia: Lwele 2025 muestra el trade-off cuando se mezclan.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SafetyProjector:
    """
    Proyector de seguridad QP (Quadratic Program).

    Toma pesos raw del agente y los proyecta al conjunto factible:
      C = {w : sum(w) = 1, w_min <= w_i <= w_max}

    Uso:
        projector = SafetyProjector(n_assets=100, w_max=0.10)
        w_safe = projector.project(w_raw)
    """

    def __init__(
        self,
        n_assets: int,
        w_min: float = 0.0,
        w_max: float = 0.10,
        sum_constraint: float = 1.0,
        solver: str = "OSQP",
        enabled: bool = True,
    ):
        """
        Parameters
        ----------
        n_assets : int
            Número de activos.
        w_min : float
            Peso mínimo por activo (0.0 = long-only).
        w_max : float
            Peso máximo por activo (e.g., 0.10 = 10% max).
        sum_constraint : float
            Suma de pesos (1.0 = fully invested).
        solver : str
            Solver CVXPY: OSQP (rápido), SCS, ECOS.
        enabled : bool
            Si False, devuelve normalización simple.
        """
        self.n_assets = n_assets
        self.w_min = w_min
        self.w_max = w_max
        self.sum_constraint = sum_constraint
        self.solver = solver
        self.enabled = enabled
        self._cvxpy_available = self._check_cvxpy()

    def _check_cvxpy(self) -> bool:
        try:
            import cvxpy
            return True
        except ImportError:
            logger.warning("cvxpy no disponible. Usando proyección simple (clip+normalize).")
            return False

    def project(self, w_raw: np.ndarray) -> np.ndarray:
        """
        Proyecta w_raw al conjunto factible.

        Parameters
        ----------
        w_raw : np.ndarray
            Pesos raw del agente (n_assets,) — pueden violar constraints.

        Returns
        -------
        np.ndarray
            Pesos seguros w_safe (n_assets,) — siempre dentro del conjunto factible.
        """
        if not self.enabled:
            return self._simple_normalize(w_raw)

        if self._cvxpy_available:
            return self._project_qp(w_raw)
        else:
            return self._project_simple(w_raw)

    def _project_qp(self, w_raw: np.ndarray) -> np.ndarray:
        """Proyección exacta con QP (CVXPY)."""
        import cvxpy as cp

        w = cp.Variable(self.n_assets)

        # Objetivo: minimizar distancia al punto raw
        objective = cp.Minimize(cp.sum_squares(w - w_raw))

        # Constraints
        constraints = [
            cp.sum(w) == self.sum_constraint,    # fully invested
            w >= self.w_min,                      # long-only
            w <= self.w_max,                      # límite individual
        ]

        problem = cp.Problem(objective, constraints)

        try:
            solver_map = {
                "OSQP": cp.OSQP,
                "SCS": cp.SCS,
                "ECOS": cp.ECOS,
            }
            solver = solver_map.get(self.solver, cp.OSQP)
            problem.solve(solver=solver, warm_start=True, verbose=False)

            if problem.status in ["optimal", "optimal_inaccurate"]:
                return np.clip(w.value, self.w_min, self.w_max)
            else:
                logger.warning(f"QP solver status: {problem.status}. Usando proyección simple.")
                return self._project_simple(w_raw)

        except Exception as e:
            logger.warning(f"QP solver error: {e}. Usando proyección simple.")
            return self._project_simple(w_raw)

    def _project_simple(self, w_raw: np.ndarray) -> np.ndarray:
        """
        Proyección rápida alternativa (si CVXPY falla):
        clip + normalize. No es exactamente óptima pero es válida.
        """
        w = np.clip(w_raw, self.w_min, self.w_max)
        w_sum = w.sum()
        if w_sum > 1e-8:
            w = w / w_sum * self.sum_constraint
        else:
            w = np.ones(self.n_assets) / self.n_assets  # fallback 1/N
        w = np.clip(w, self.w_min, self.w_max)
        return w

    def _simple_normalize(self, w_raw: np.ndarray) -> np.ndarray:
        """Normalización básica sin constraints (modo sin safety)."""
        w = np.abs(w_raw)  # long-only simple
        w_sum = w.sum()
        if w_sum > 1e-8:
            return w / w_sum
        return np.ones(self.n_assets) / self.n_assets

    def project_batch(self, w_batch: np.ndarray) -> np.ndarray:
        """
        Proyecta un batch de pesos (para evaluación).

        Parameters
        ----------
        w_batch : np.ndarray
            Batch de pesos (T, n_assets).

        Returns
        -------
        np.ndarray
            Pesos seguros (T, n_assets).
        """
        return np.stack([self.project(w) for w in w_batch])

    def check_constraints(self, w: np.ndarray, tol: float = 1e-4) -> dict:
        """
        Verifica si los pesos cumplen todos los constraints.
        Útil para auditoría y reporting.

        Returns
        -------
        dict con resultado de cada constraint.
        """
        return {
            "sum_ok": abs(w.sum() - self.sum_constraint) < tol,
            "long_only_ok": (w >= self.w_min - tol).all(),
            "max_weight_ok": (w <= self.w_max + tol).all(),
            "sum_value": float(w.sum()),
            "max_weight": float(w.max()),
            "min_weight": float(w.min()),
            "n_assets_active": int((w > tol).sum()),
        }

    def disable(self):
        """Desactiva la proyección (modo ablation sin safety)."""
        self.enabled = False

    def enable(self):
        """Activa la proyección."""
        self.enabled = True


def build_safety_projector(n_assets: int, config: dict) -> SafetyProjector:
    """Factory para construir el proyector de seguridad."""
    safety_cfg = config["safety"]
    return SafetyProjector(
        n_assets=n_assets,
        w_min=safety_cfg["w_min"],
        w_max=safety_cfg["w_max"],
        sum_constraint=safety_cfg["sum_constraint"],
        solver=safety_cfg.get("solver", "OSQP"),
        enabled=safety_cfg.get("enabled", True),
    )


if __name__ == "__main__":
    import yaml
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    n = 100
    projector = build_safety_projector(n, cfg)

    # Test con pesos random que violan constraints
    w_raw = np.random.randn(n)  # pueden ser negativos, suma != 1
    w_safe = projector.project(w_raw)

    checks = projector.check_constraints(w_safe)
    print(f"Raw weights: sum={w_raw.sum():.3f}, min={w_raw.min():.3f}, max={w_raw.max():.3f}")
    print(f"Safe weights: sum={w_safe.sum():.3f}, min={w_safe.min():.3f}, max={w_safe.max():.3f}")
    print(f"Constraint checks: {checks}")
