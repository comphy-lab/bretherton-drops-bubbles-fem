"""Independent axisymmetric P2 surface geometry; no production/native imports."""
from __future__ import annotations
import math
import numpy as np


class GateFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GateFailure(message)


def checked_geometry(points, connectivity):
    p = np.asarray(points, dtype=np.float64)
    c = np.asarray(connectivity)
    require(p.ndim == 2 and p.shape[1] == 2 and len(p) >= 3, "expected z,r nodes")
    require(np.isfinite(p).all() and np.all(p[:, 1] >= 0), "invalid radial geometry")
    require(c.ndim == 2 and c.shape[1] == 3 and c.dtype.kind in "iu", "expected P2 integer connectivity")
    require(len(c) > 0 and np.all(c >= 0) and np.all(c < len(p)), "connectivity bounds")
    require(all(len(set(row)) == 3 for row in c.tolist()), "duplicate local P2 nodes")
    require(np.array_equal(c[1:, 0], c[:-1, 2]), "disconnected or inconsistently oriented chain")
    require(len(set(c[:, 1])) == len(c), "reused midside node")
    require(not set(c[:, 1]) & set(c[:, (0, 2)].ravel()), "midside node is an endpoint")
    require(len(set(c.ravel())) == 2 * len(c) + 1, "repeated chain endpoint")
    require(set(c.ravel()) == set(range(len(p))), "unused node")
    # Direction is retained, never silently flipped. A reversed chain has negative volume.
    return p, c.astype(np.int64)


def quadrature():
    x = np.array([-math.sqrt(3 / 5), 0., math.sqrt(3 / 5)])
    w = np.array([5 / 9, 8 / 9, 5 / 9])
    n = np.column_stack((x * (x - 1) / 2, 1 - x*x, x * (x + 1) / 2))
    dn = np.column_stack((x - .5, -2*x, x + .5))
    return w, n, dn


def volume_differential(points, connectivity, displacement):
    """Return signed V, complex dV, and an absolute integrand scale.

    Coordinates and displacement columns are (z,r). Three-point Gauss is exact
    for these degree-five polynomials; it is not exact for an analytic sphere
    represented approximately by P2 elements. No normal projection is used.
    """
    p, c = checked_geometry(points, connectivity)
    d = np.asarray(displacement, dtype=np.complex128)
    require(d.shape == p.shape and np.isfinite(d).all(), "invalid displacement")
    w, n, dn = quadrature()
    zprime = p[c, 0] @ dn.T
    r = p[c, 1] @ n.T
    require(np.all(r >= 0), 'P2 radius became negative at quadrature points')
    dr = d[c, 1] @ n.T
    dzprime = d[c, 0] @ dn.T
    first, second = 2*r*dr*zprime, r*r*dzprime
    dv = math.pi * np.sum((first + second) * w)
    scale = float(math.pi * np.sum((np.abs(first) + np.abs(second)) * w))
    v = float(math.pi * np.sum(r*r*zprime*w))
    require(np.isfinite([v, dv.real, dv.imag, scale]).all(), "nonfinite quadrature")
    return {"signed_volume": v, "delta_volume": complex(dv), "absolute_integrand_scale": scale,
            "normalised_absolute_delta_volume": abs(dv)/scale if scale else None}


def complex_record(value):
    return {"real": float(value.real), "imag": float(value.imag), "absolute": float(abs(value))}


def compare_row(result, row_action, alpha, *, row_factor):
    """row_factor must be source-established, never fitted to this result."""
    require(np.isfinite(row_factor) and row_factor != 0, "unverified row factor")
    expected = row_factor * result["delta_volume"]
    difference = complex(row_action) - expected
    scale = abs(row_factor)*result["absolute_integrand_scale"]
    return {"geometric": complex_record(result["delta_volume"]),
            "stored_J_row_action": complex_record(row_action), "alpha": complex_record(alpha),
            "row_factor": row_factor, "discrepancy": complex_record(difference),
            "scale_normalised_discrepancy": abs(difference)/scale if scale else None,
            "acceptance": "diagnostic only; no new pass allowance"}


def volume_coefficients(points, connectivity):
    """Independent analytic P2 derivative coefficients in (z,r) ordinal order."""
    p,c=checked_geometry(points,connectivity)
    w,n,dn=quadrature()
    r=p[c,1] @ n.T
    zp=p[c,0] @ dn.T
    coeff=np.zeros_like(p)
    for local in range(3):
        np.add.at(coeff[:,0],c[:,local],math.pi*np.sum(r*r*dn[:,local]*w,axis=1))
        np.add.at(coeff[:,1],c[:,local],2*math.pi*np.sum(r*zp*n[:,local]*w,axis=1))
    return coeff
