"""Per-vertex principal curvature of a triangle mesh, vectorized in NumPy.

The Curved export uses it to split freeform patches where the curvature
changes character (ridge, valley, inflection) instead of where normals merely
differ. Per vertex: a least-squares osculating paraboloid `z = a x^2 + b xy +
c y^2` over the 1-ring in a normal frame; the eigenvalues of [[2a, b], [b, 2c]]
are the principal curvatures (1/mm for millimetre input). Edge sums use
`np.add.at` and one batched 3x3 solve.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def vertex_normals(verts: np.ndarray,
                   faces: Sequence[tuple[int, int, int]]) -> np.ndarray:
    """Area-weighted vertex normals (the cross product's own magnitude is
    twice the triangle area, so summing raw cross products already weights
    by area)."""
    f = np.asarray(faces, dtype=np.int64)
    n = np.cross(verts[f[:, 1]] - verts[f[:, 0]],
                 verts[f[:, 2]] - verts[f[:, 0]])
    out = np.zeros_like(verts)
    for k in range(3):
        np.add.at(out, f[:, k], n)
    ln = np.linalg.norm(out, axis=1)
    ln[ln < 1e-20] = 1.0
    return out / ln[:, None]


def _tangent_frames(normals: np.ndarray):
    """Two orthonormal tangents per normal, chosen without a branch that
    could flip between neighbouring vertices."""
    ref = np.zeros_like(normals)
    small = np.abs(normals[:, 0]) < 0.9
    ref[small, 0] = 1.0
    ref[~small, 1] = 1.0
    t1 = np.cross(normals, ref)
    ln = np.linalg.norm(t1, axis=1)
    ln[ln < 1e-20] = 1.0
    t1 = t1 / ln[:, None]
    t2 = np.cross(normals, t1)
    return t1, t2


def principal_curvatures(verts: np.ndarray,
                         faces: Sequence[tuple[int, int, int]],
                         smooth_passes: int = 1):
    """Return `(k_min, k_max, normals)` per vertex, k_min <= k_max.

    The sign follows the mesh's normal orientation. `smooth_passes` averages
    each vertex with its 1-ring afterwards, which keeps the shape index stable
    on coarse meshes."""
    f = np.asarray(faces, dtype=np.int64)
    nv = len(verts)
    normals = vertex_normals(verts, faces)
    t1, t2 = _tangent_frames(normals)

    # every directed edge of every triangle, both ways
    e0 = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    e1 = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    d = verts[e1] - verts[e0]
    x = np.einsum("ij,ij->i", d, t1[e0])
    y = np.einsum("ij,ij->i", d, t2[e0])
    z = np.einsum("ij,ij->i", d, normals[e0])
    # normalize by the edge length so long and short edges weigh the same;
    # otherwise one long edge dominates a vertex's whole fit
    scale = np.maximum(np.sqrt(x * x + y * y), 1e-12)
    w = 1.0 / scale
    basis = np.stack([x * x, x * y, y * y], axis=1) * w[:, None]
    rhs = z * w

    AtA = np.zeros((nv, 3, 3))
    Atz = np.zeros((nv, 3))
    for i in range(3):
        for j in range(3):
            np.add.at(AtA[:, i, j], e0, basis[:, i] * basis[:, j])
        np.add.at(Atz[:, i], e0, basis[:, i] * rhs)
    # tiny ridge term: a vertex with too few or collinear neighbours would
    # otherwise give a singular system (and curvature 0 is the right answer
    # there anyway)
    AtA += np.eye(3) * 1e-12
    try:
        sol = np.linalg.solve(AtA, Atz[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        sol = np.zeros((nv, 3))
    sol = np.nan_to_num(sol)

    a, b, c = sol[:, 0], sol[:, 1], sol[:, 2]
    # eigenvalues of [[2a, b], [b, 2c]]
    tr = a + c                      # (2a + 2c) / 2
    det_term = np.sqrt(np.maximum((a - c) ** 2 + b * b, 0.0))
    k_max = tr + det_term
    k_min = tr - det_term

    for _ in range(max(smooth_passes, 0)):
        k_max = _smooth(k_max, e0, e1, nv)
        k_min = _smooth(k_min, e0, e1, nv)
    return k_min, k_max, normals


def _smooth(values: np.ndarray, e0: np.ndarray, e1: np.ndarray, nv: int):
    acc = np.zeros(nv)
    cnt = np.zeros(nv)
    np.add.at(acc, e0, values[e1])
    np.add.at(cnt, e0, 1.0)
    cnt[cnt == 0] = 1.0
    return 0.5 * values + 0.5 * (acc / cnt)


def shape_index(k_min: np.ndarray, k_max: np.ndarray) -> np.ndarray:
    """Koenderink & van Doorn shape index in [-1, 1]: -1 cup, -0.5 rut,
    0 saddle, +0.5 ridge, +1 cap. Scale-free; 0 where flat or umbilic."""
    denom = k_max - k_min
    out = np.zeros_like(k_min)
    ok = np.abs(denom) > 1e-12
    out[ok] = (2.0 / math.pi) * np.arctan((k_max[ok] + k_min[ok]) / denom[ok])
    return out


def face_shape_index(verts: np.ndarray,
                     faces: Sequence[tuple[int, int, int]],
                     smooth_passes: int = 1) -> np.ndarray:
    """Per-TRIANGLE shape index, averaged from its three vertices -- the form
    the segmentation actually consumes."""
    k_min, k_max, _ = principal_curvatures(verts, faces, smooth_passes)
    si = shape_index(k_min, k_max)
    f = np.asarray(faces, dtype=np.int64)
    return si[f].mean(axis=1)
