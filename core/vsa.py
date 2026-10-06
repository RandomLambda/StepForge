"""Variational shape approximation: cut a mesh into a chosen number of
compact regions.

Cohen-Steiner, Alliez & Desbrun, "Variational Shape Approximation" (SIGGRAPH
2004): Lloyd's algorithm on the mesh, alternating (a) flooding outward from
each proxy's seed in order of increasing error and (b) refitting each proxy to
its region. The flooding order makes the regions compact, which plain
normal-cone region growing does not: grown regions come out convoluted and
cannot be flattened onto a square without distortion. Distortion metric: the
paper's L^2,1 (area-weighted squared normal difference). NumPy plus a heap.
"""
from __future__ import annotations

import heapq
from collections.abc import Sequence

import numpy as np


def _face_data(verts: np.ndarray, faces, face_idx: Sequence[int]):
    f = np.array([faces[i] for i in face_idx], dtype=np.int64)
    v0, v1, v2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area2 = np.linalg.norm(cross, axis=1)
    area = 0.5 * area2
    safe = np.maximum(area2, 1e-20)
    normals = cross / safe[:, None]
    centroids = (v0 + v1 + v2) / 3.0
    return normals, area, centroids


def _adjacency(faces, face_idx: Sequence[int]) -> dict[int, list[int]]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for f in face_idx:
        a, b, c = faces[f]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_faces.setdefault(key, []).append(f)
    adj: dict[int, list[int]] = {f: [] for f in face_idx}
    for fs in edge_faces.values():
        if len(fs) == 2:
            i, j = fs
            adj[i].append(j)
            adj[j].append(i)
    return adj


def partition(verts: np.ndarray, faces, face_idx: Sequence[int], k: int,
              iterations: int = 12,
              seeds: Sequence[int] | None = None) -> list[list[int]]:
    """Split `face_idx` into (at most) `k` compact, connected regions.

    Returns the regions as lists of triangle indices, largest first, with
    every input triangle in exactly one of them. Regions can come back empty
    if two proxies collapse onto each other; those are dropped."""
    face_idx = list(face_idx)
    n = len(face_idx)
    if n == 0:
        return []
    k = max(1, min(int(k), n))
    if k == 1:
        return [face_idx]

    local = {f: i for i, f in enumerate(face_idx)}
    normals, area, centroids = _face_data(verts, faces, face_idx)
    adj = _adjacency(faces, face_idx)

    if seeds is None:
        # Spread the initial seeds out rather than taking the first k
        # triangles: farthest-point sampling on the centroids, which starts
        # Lloyd's iteration somewhere it can actually converge from.
        seeds_local = [0]
        d = np.linalg.norm(centroids - centroids[0], axis=1)
        for _ in range(k - 1):
            j = int(np.argmax(d))
            seeds_local.append(j)
            d = np.minimum(d, np.linalg.norm(centroids - centroids[j], axis=1))
    else:
        seeds_local = [local[s] for s in seeds if s in local][:k]
    proxy_n = normals[seeds_local].copy()

    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(max(iterations, 1)):
        labels[:] = -1
        heap = []
        for r, s in enumerate(seeds_local):
            labels[s] = r
            for nb in adj[face_idx[s]]:
                j = local[nb]
                heapq.heappush(heap, (_dist(normals[j], area[j], proxy_n[r]),
                                      j, r))
        while heap:
            _e, j, r = heapq.heappop(heap)
            if labels[j] != -1:
                continue
            labels[j] = r
            for nb in adj[face_idx[j]]:
                m = local[nb]
                if labels[m] == -1:
                    heapq.heappush(heap, (_dist(normals[m], area[m], proxy_n[r]),
                                          m, r))
        # any triangle no seed could reach (a disconnected island) starts its
        # own region rather than being dropped
        for j in range(n):
            if labels[j] == -1:
                labels[j] = len(proxy_n)
                proxy_n = np.vstack([proxy_n, normals[j][None, :]])
                seeds_local.append(j)

        # refit each proxy to its region, and re-seed it at its best triangle
        changed = False
        for r in range(len(proxy_n)):
            members = np.nonzero(labels == r)[0]
            if not len(members):
                continue
            acc = (normals[members] * area[members, None]).sum(axis=0)
            ln = float(np.linalg.norm(acc))
            if ln > 1e-20:
                proxy_n[r] = acc / ln
            errs = _dist_many(normals[members], area[members], proxy_n[r])
            best = int(members[int(np.argmin(errs))])
            if best != seeds_local[r]:
                seeds_local[r] = best
                changed = True
        if not changed:
            break

    out: list[list[int]] = []
    for r in range(len(proxy_n)):
        members = [face_idx[j] for j in np.nonzero(labels == r)[0]]
        if members:
            out.append(members)
    out.sort(key=len, reverse=True)
    return out


def _dist(normal, a, proxy_n) -> float:
    d = normal - proxy_n
    return float(a * (d @ d))


def _dist_many(normals, areas, proxy_n) -> np.ndarray:
    d = normals - proxy_n
    return areas * np.einsum("ij,ij->i", d, d)
