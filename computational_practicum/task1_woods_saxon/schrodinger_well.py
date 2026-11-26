"""Bound states of a one-dimensional finite well with arbitrary U(x).

Units
-----
hbar^2 / (2m) = 1, so the stationary Schrodinger equation reads

    -psi''(x) + U(x) psi(x) = E psi(x).

Lengths are measured in units of the well width a, energies in units of
hbar^2 / (2 m a^2).  The well is centred at the origin and occupies
|x| <= a/2; outside it U = 0 and bound states decay as exp(-kappa|x|),
kappa = sqrt(-E).

Two independent numerical methods are implemented:

* ``solve_fd`` - finite differences on a cell-centred uniform grid.  The
  Hamiltonian is a symmetric tridiagonal matrix and all E < 0 are obtained
  in one call.  Cell centring puts the well boundary exactly on a cell
  face, keeping the scheme second-order accurate for a steep edge.
* ``numerov_levels`` - Numerov integration (fourth order) from the origin
  and from the outer boundary, matched at an interior point.  Matching is
  done with the cross-multiplied Wronskian, which stays bounded, so the
  method is stable for deeply bound states where a one-sided integration
  into the forbidden region would be exponentially ill-conditioned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq


@dataclass
class BoundState:
    """A normalized bound state on a cell-centred grid."""

    energy: float
    psi: np.ndarray
    nodes: int

    @property
    def parity(self) -> str:
        return "even" if self.nodes % 2 == 0 else "odd"


def woods_saxon(x, depth, radius=0.5, diffuseness=0.05):
    """Woods-Saxon well centred at the origin, defined on the whole line.

    U(x) = -depth / (1 + exp((|x| - radius)/diffuseness)).

    ``radius`` is the half-depth radius, so the width at half depth is
    2*radius.  The potential has smooth exponential tails and no artificial
    truncation, which keeps both numerical methods spectrally clean.
    """
    x = np.asarray(x, dtype=float)
    return -depth / (1.0 + np.exp((np.abs(x) - radius) / diffuseness))


def poschl_teller(x, alpha, lam):
    """Poschl-Teller well U = -alpha^2 lam(lam+1)/cosh^2(alpha x).

    Has exactly ``ceil(lam)`` bound states with E_n = -alpha^2 (lam - n)^2.
    """
    return -alpha**2 * lam * (lam + 1.0) / np.cosh(alpha * np.asarray(x)) ** 2


def poschl_teller_levels(alpha, lam):
    """Exact bound-state energies of the Poschl-Teller well."""
    n = np.arange(int(np.ceil(lam)))
    return np.sort(-(alpha**2) * (lam - n) ** 2)


def _count_nodes(psi):
    scale = np.max(np.abs(psi))
    values = psi[np.abs(psi) > 1.0e-10 * scale]
    return int(np.count_nonzero(np.diff(np.sign(values))))


def solve_fd(potential, half_width, n_intervals):
    """Solve -psi'' + U psi = E psi by finite differences.

    The grid is cell-centred, x_j = -L + (j + 1/2)h, so the edges |x| = L and
    any interface at |x| = a/2 fall on cell faces.  Dirichlet conditions
    psi(+-L) = 0 are imposed through ghost cells.  Returns the grid and the
    bound states sorted by energy.
    """
    h = 2.0 * half_width / n_intervals
    x = -half_width + (np.arange(n_intervals) + 0.5) * h
    diagonal = 2.0 / h**2 + potential(x)
    diagonal[0] += 1.0 / h**2
    diagonal[-1] += 1.0 / h**2
    offdiagonal = np.full(n_intervals - 1, -1.0 / h**2)
    eigenvalues, eigenvectors = eigh_tridiagonal(
        diagonal, offdiagonal, select="v", select_range=(-1.0e9, 0.0)
    )

    states = []
    for energy, vector in zip(eigenvalues, eigenvectors.T):
        psi = vector / np.sqrt(h * np.sum(vector**2))
        nodes = _count_nodes(psi)
        if psi[np.argmax(np.abs(psi))] < 0.0:
            psi = -psi
        states.append(BoundState(float(energy), psi, nodes))
    states.sort(key=lambda state: state.energy)
    return x, states


def _numerov_outward(energy, potential, x_match, h, parity):
    """Integrate from the origin to x_match (oscillatory region)."""
    n = int(round(x_match / h))
    x = np.arange(n + 1) * h
    f = potential(x) - energy
    psi = np.empty(n + 1)
    c = h * h / 12.0
    if parity == "even":
        psi[0] = 1.0
        psi[1] = 1.0 + 0.5 * h * h * f[0]
    elif parity == "odd":
        psi[0] = 0.0
        psi[1] = h
    else:
        raise ValueError("parity must be 'even' or 'odd'")
    for j in range(1, n):
        psi[j + 1] = (
            (2.0 + 10.0 * c * f[j]) * psi[j]
            - (1.0 - c * f[j - 1]) * psi[j - 1]
        ) / (1.0 - c * f[j + 1])
    return x, psi


def _numerov_inward(energy, potential, x_match, x_max, h):
    """Integrate from x_max down to x_match (forbidden region, stable sense).

    The two starting values follow the decaying tail
    psi(x_max - h) = psi(x_max) exp(kappa h); starting instead from
    psi(x_max) = 0 seeds the exponentially growing (unphysical) solution,
    which is many orders of magnitude larger at x_max and swamps the tail.
    """
    m = int(round((x_max - x_match) / h))
    y = x_max - np.arange(m + 1) * h
    f = potential(y) - energy
    psi = np.empty(m + 1)
    c = h * h / 12.0
    kappa = np.sqrt(-energy)
    psi[0] = 1.0
    psi[1] = np.exp(kappa * h)
    for j in range(1, m):
        psi[j + 1] = (
            (2.0 + 10.0 * c * f[j]) * psi[j]
            - (1.0 - c * f[j - 1]) * psi[j - 1]
        ) / (1.0 - c * f[j + 1])
    return y, psi


def _numerov_residual(energy, potential, x_match, x_max, h, parity):
    """Cross-multiplied Wronskian of the two solutions at x_match.

    Zero when the logarithmic derivatives coincide, i.e. at an eigenvalue.
    """
    _, u = _numerov_outward(energy, potential, x_match, h, parity)
    _, v = _numerov_inward(energy, potential, x_match, x_max, h)
    u = u / np.max(np.abs(u))
    v = v / np.max(np.abs(v))
    du = (11.0 * u[-1] - 18.0 * u[-2] + 9.0 * u[-3] - 2.0 * u[-4]) / (6.0 * h)
    # v is ordered with decreasing x, so the one-sided stencil flips sign
    dv = (-11.0 * v[-1] + 18.0 * v[-2] - 9.0 * v[-3] + 2.0 * v[-4]) / (6.0 * h)
    return du * v[-1] - dv * u[-1]


def numerov_levels(potential, depth, x_match, x_max, h=0.001, dE=0.5):
    """All bound-state energies from two-sided Numerov shooting.

    Even and odd sectors are scanned separately; sign changes of the
    Wronskian residual bracket the eigenvalues and are refined by brentq.
    """
    grid = np.arange(-depth + 1.0e-6, -1.0e-9, dE)
    levels = []
    for parity in ("even", "odd"):
        residual = np.array(
            [
                _numerov_residual(E, potential, x_match, x_max, h, parity)
                for E in grid
            ]
        )
        for i in range(grid.size - 1):
            if residual[i] * residual[i + 1] < 0.0:
                root = brentq(
                    lambda E: _numerov_residual(
                        E, potential, x_match, x_max, h, parity
                    ),
                    grid[i],
                    grid[i + 1],
                    xtol=1.0e-12,
                    rtol=1.0e-15,
                    maxiter=200,
                )
                levels.append(root)
    levels.sort()
    return np.array(levels)


def numerov_wavefunction(energy, potential, x_match, x_max, h, parity):
    """Matched Numerov wavefunction on [-x_max, x_max], normalized to 1.

    Returns x, psi with psi(+-x_max) ~ 0 and parity fixed by ``parity``.
    """
    x_out, u = _numerov_outward(energy, potential, x_match, h, parity)
    y_in, v = _numerov_inward(energy, potential, x_match, x_max, h)
    v *= u[-1] / v[-1]
    x_half = np.concatenate([x_out, y_in[1:][::-1]])
    psi_half = np.concatenate([u, v[1:][::-1]])
    if parity == "even":
        x = np.concatenate([-x_half[:0:-1], x_half])
        psi = np.concatenate([psi_half[:0:-1], psi_half])
    else:
        x = np.concatenate([-x_half[::-1], x_half])
        psi = np.concatenate([-psi_half[::-1], psi_half])
    psi /= np.sqrt(np.trapezoid(psi**2, x))
    return x, psi


def tabulate_potential(potential, width, n_points):
    """Sample U on [0, width] and build a cubic-spline interpolant.

    Returns the table (x, U), the interpolant callable defined on the whole
    line (zero outside the well), and the spline object.
    """
    x_table = np.linspace(0.0, width, n_points)
    u_table = potential(x_table - 0.5 * width)
    spline = CubicSpline(x_table, u_table)

    def u_interpolated(x):
        x = np.asarray(x, dtype=float)
        values = np.zeros_like(x)
        inside = np.abs(x) <= 0.5 * width
        values[inside] = spline(x[inside] + 0.5 * width)
        return values

    return x_table, u_table, u_interpolated, spline
