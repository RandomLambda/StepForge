"""One shared, simplified boundary between every pair of neighbouring faces.

Faces that simplify their own boundaries independently keep different
vertices along the same physical edge, so they stop referencing the same
`EDGE_CURVE` (keyed by vertex pair) and the shell opens there: a T-junction,
an edge used by one face instead of two. That was the source of spikes, cracks
and non-watertight re-imports in Curved mode (every `EDGE_CURVE` of a
`CLOSED_SHELL` must be used exactly twice; `tests/step_topology.py` checks it).

Given every face group up front (primitive patches, freeform regions, merged
flat groups, single-triangle leftovers), the mesh edges between two different
groups are collected and split into **chains**: maximal runs of boundary edge
with the same pair of groups on either side, ending where three or more groups
meet (a "node"). A chain is simplified exactly once and both faces that share
it get the same vertex sequence, so they reference the same edges by
construction. The simplification is Douglas-Peucker bounded by the export
tolerance, or, where a chain borders a fitted cylinder, cone or torus, the
merge of points that lie on one circle of that surface.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import fit as _fit

# How much tighter a boundary chain is simplified when one of the two faces it
# separates is curved (see `_register`). 1.0 means no tighter: tightening grew
# the file by 30% without reducing sliver triangles, whose cause is the
# boundary/interior sampling-density mismatch handled in
# `convert._register_edge_hints`.
CURVED_CHAIN_TIGHTEN = 1.0


def _dp_mark(pts: np.ndarray, i: int, j: int, keep: list[bool], tol: float):
    """Douglas-Peucker: keep every point between i and j that is further than
    `tol` from the chord i->j."""
    if j <= i + 1:
        return
    a = pts[i]
    ab = pts[j] - a
    ab_len = float(np.linalg.norm(ab))
    seg = pts[i + 1:j] - a
    if ab_len < 1e-12:
        d = np.linalg.norm(seg, axis=1)
    else:
        t = np.clip((seg @ ab) / (ab_len * ab_len), 0.0, 1.0)
        d = np.linalg.norm(seg - t[:, None] * ab, axis=1)
    k = int(np.argmax(d))
    if d[k] <= tol:
        return
    split = i + 1 + k
    keep[split] = True
    _dp_mark(pts, i, split, keep, tol)
    _dp_mark(pts, split, j, keep, tol)


def simplify_chain(chain: list[int], verts: np.ndarray, tolerance: float,
                   arc_fit=None) -> list[int]:
    """Simplify one open chain of mesh vertices, endpoints always kept.

    `arc_fit`, if given, is a fitted cylinder, cone or torus on one side of
    this chain: interior points whose neighbouring edges both classify as
    CIRCLE on the same circle of that fit (`fit._same_circle`) are dropped
    first, since they lie on one exact circle and the face can write one long
    arc. Douglas-Peucker then handles what is left; it is error-bounded by
    `tolerance`, unlike an angle-only collinearity test, which can walk a
    gently curved boundary arbitrarily far from where it started."""
    n = len(chain)
    if n <= 2:
        return list(chain)

    if arc_fit is not None and getattr(arc_fit, "kind", None) in ("cylinder", "cone", "torus"):
        kept = [chain[0]]
        for idx in range(1, n - 1):
            prev, cur, nxt = chain[idx - 1], chain[idx], chain[idx + 1]
            s1 = _fit.classify_edge(arc_fit, verts[prev], verts[cur])
            s2 = _fit.classify_edge(arc_fit, verts[cur], verts[nxt])
            if (s1.kind == "circle" and s2.kind == "circle"
                    and _fit._same_circle(s1, s2, tolerance)):
                continue
            kept.append(cur)
        kept.append(chain[-1])
        chain = kept
        n = len(chain)
        if n <= 2:
            return chain

    pts = verts[chain]
    keep = [False] * n
    keep[0] = keep[-1] = True
    _dp_mark(pts, 0, n - 1, keep, tolerance)
    return [chain[i] for i in range(n) if keep[i]]


def _spans_for_chain(chain: list[int], simp: list[int]) -> dict[tuple[int, int], list[int]]:
    """Recover, for each edge of the simplified chain `simp`, the raw
    mesh-vertex run between its two endpoints that simplification collapsed
    into that edge.

    `simplify_chain` and `simplify_ring` only drop points, so `simp` is a
    subsequence of `chain` in the same direction and one linear scan recovers
    each run. Keyed in both directions, so a caller need not know which way a
    face's loop travels the chain. The writer (`brep_export._FaceBuilder`)
    uses it to fit a curve through a curved stretch instead of the single
    straight LINE a bare two-point edge would get."""
    spans: dict[tuple[int, int], list[int]] = {}
    if len(simp) < 2:
        return spans
    k = 0
    start = 0
    for i in range(1, len(chain)):
        if chain[i] == simp[k + 1]:
            a, b = simp[k], simp[k + 1]
            run = chain[start:i + 1]
            spans[(a, b)] = run
            spans[(b, a)] = list(reversed(run))
            k += 1
            start = i
            if k + 1 >= len(simp):
                break
    return spans


def simplify_ring(ring: list[int], verts: np.ndarray, tolerance: float,
                  arc_fit=None) -> list[int]:
    """Simplify a closed chain (a ring with no node on it, e.g. a bore rim
    shared by exactly two faces): cut at the first vertex and its most distant
    partner so neither half starts out degenerate, then simplify the two
    halves as open chains.

    A ring shared by two faces is never collapsed below 3 points, even when
    every edge classifies as the same CIRCLE: with 2 points both faces would
    reference the same undirected (A, B) pair twice, giving one EDGE_CURVE four
    uses instead of the two a closed shell needs (`_resolve_collisions` is the
    backstop). A seamed loop's own internal pinch connector, which only one
    face touches, is a different and safe case of an edge used twice (see
    `brep_export._make_seamed_loop`)."""
    n = len(ring)
    if n <= 3:
        return list(ring)
    pts = verts[ring]
    far = int(np.argmax(np.linalg.norm(pts - pts[0], axis=1)))
    if far == 0:
        return list(ring)
    first = simplify_chain(ring[:far + 1], verts, tolerance, arc_fit)
    second = simplify_chain(ring[far:] + [ring[0]], verts, tolerance, arc_fit)
    out = first[:-1] + second[:-1]
    return out if len(out) >= 3 else list(ring)


class BoundaryGraph:
    """Shared boundary chains for one solid's complete set of face groups.

    Build it once, after every triangle has been assigned to a group, then
    call `simplify_loop` for each face's raw boundary loop. Groups are
    identified by their index in the `groups` list passed to the
    constructor."""

    def __init__(self, verts: np.ndarray,
                 faces: Sequence[tuple[int, int, int]],
                 groups: Sequence[Sequence[int]],
                 tolerance: float,
                 arc_fits: dict[int, object] | None = None,
                 curved_gids: set | None = None,
                 curved_factor: float = CURVED_CHAIN_TIGHTEN):
        self.verts = verts
        self.tolerance = tolerance
        arc_fits = arc_fits or {}
        self._curved = curved_gids or set()
        self._curved_factor = curved_factor

        group_of: dict[int, int] = {}
        for gid, g in enumerate(groups):
            for f in g:
                group_of[f] = gid

        # undirected mesh edge -> the (up to two) groups touching it
        edge_groups: dict[tuple[int, int], set] = {}
        for f, gid in group_of.items():
            a, b, c = faces[f]
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                edge_groups.setdefault(key, set()).add(gid)

        # A boundary edge separates two different groups (or a group from a
        # hole in the mesh, which shows up as a single-group edge used by
        # only one triangle).
        tri_count: dict[tuple[int, int], int] = {}
        for f in group_of:
            a, b, c = faces[f]
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                tri_count[key] = tri_count.get(key, 0) + 1

        self.label: dict[tuple[int, int], frozenset] = {}
        adj: dict[int, list[tuple[int, int]]] = {}
        for key, gids in edge_groups.items():
            if len(gids) == 1 and tri_count.get(key, 0) == 2:
                continue  # interior to one group
            lab = frozenset(gids)
            self.label[key] = lab
            adj.setdefault(key[0], []).append(key)
            adj.setdefault(key[1], []).append(key)
        self._adj = adj

        # A vertex is a node when its boundary edges do not form one single
        # pass-through (degree 2 with the same pair of groups on both sides).
        self._nodes = set()
        for v, edges in adj.items():
            if len(edges) != 2 or self.label[edges[0]] != self.label[edges[1]]:
                self._nodes.add(v)

        self.fallbacks = 0   # loops that could not be rewritten (diagnostic)
        self._chain_of_edge: dict[tuple[int, int], int] = {}
        self._chains: list[list[int]] = []
        self._simplified: list[list[int]] = []
        self._build_chains(arc_fits)
        self._resolve_collisions()
        self._protect_small_faces(faces, groups)
        self._protect_crossing_faces(faces, groups)
        # Again after the protection passes: restoring a chain's raw vertices
        # can re-create a collision. Raw chains are disjoint runs of real mesh
        # edges, so this converges immediately.
        self._resolve_collisions()

    # -- construction -------------------------------------------------
    def _other(self, edge: tuple[int, int], v: int) -> int:
        return edge[1] if edge[0] == v else edge[0]

    def _walk(self, start: int, first_edge: tuple[int, int]) -> list[int]:
        chain = [start]
        edge = first_edge
        cur = self._other(edge, start)
        used = [edge]
        while True:
            chain.append(cur)
            if cur in self._nodes or cur == start:
                break
            nxt = [e for e in self._adj[cur] if e != edge]
            if len(nxt) != 1:
                break
            edge = nxt[0]
            used.append(edge)
            cur = self._other(edge, cur)
        return chain, used

    def _register(self, chain: list[int], used, arc_fits: dict[int, object],
                  closed: bool):
        cid = len(self._chains)
        self._chains.append(chain)
        for e in used:
            self._chain_of_edge[e] = cid
        lab = self.label[used[0]]
        # A chain bordering a curved face is simplified more tightly (by
        # `curved_factor`): the face's boundary is written as straight LINE
        # segments between mesh vertices, so each chord cuts a corner of the
        # curved surface, and a reader trims the face by projecting the chord
        # back onto it. The further the chord from the surface, the more the
        # trim boundary wanders, which shows up as spikes in a re-imported
        # freeform face.
        tol = self.tolerance
        if any(g in self._curved for g in lab):
            tol *= self._curved_factor
        arc_fit = None
        for gid in lab:
            f = arc_fits.get(gid)
            if f is not None and getattr(f, "kind", None) in ("cylinder", "cone", "torus"):
                arc_fit = f
                break
        if closed:
            simp = simplify_ring(chain[:-1], self.verts, tol, arc_fit)
            simp = simp + [simp[0]]
        else:
            simp = simplify_chain(chain, self.verts, tol, arc_fit)
        self._simplified.append(simp)

    def _build_chains(self, arc_fits):
        seen_edges = set()
        # open chains first: they start and end at nodes
        for v in sorted(self._nodes):
            for e in self._adj.get(v, ()):
                if e in seen_edges:
                    continue
                chain, used = self._walk(v, e)
                for u in used:
                    seen_edges.add(u)
                self._register(chain, used, arc_fits, closed=False)
        # whatever is left is a closed ring with no node on it at all
        for e in self.label:
            if e in seen_edges:
                continue
            start = e[0]
            chain, used = self._walk(start, e)
            for u in used:
                seen_edges.add(u)
            self._register(chain, used, arc_fits, closed=True)

    def _resolve_collisions(self):
        """Un-simplify any two chains that collapsed onto the same vertex pair.

        Two chains can share both endpoints (two ways round a small feature);
        if both simplify to that single segment they become one `EDGE_CURVE`
        referenced by four faces instead of two. Falling back to the raw mesh
        vertices for the colliding chains always resolves it: raw chains are
        disjoint sequences of real mesh edges."""
        owner: dict[tuple[int, int], int] = {}
        conflicted = set()
        for cid, simp in enumerate(self._simplified):
            seen_here = set()
            for i in range(len(simp) - 1):
                a, b = simp[i], simp[i + 1]
                key = (a, b) if a < b else (b, a)
                if key in seen_here:
                    # The chain uses the same undirected edge twice (a ring
                    # that simplified to a there-and-back pair): both faces
                    # would reference that EDGE_CURVE twice.
                    conflicted.add(cid)
                seen_here.add(key)
                prev = owner.get(key)
                if prev is not None and prev != cid:
                    conflicted.add(prev)
                    conflicted.add(cid)
                else:
                    owner[key] = cid
        for cid in conflicted:
            self._simplified[cid] = list(self._chains[cid])
        self.collisions = len(conflicted)

    def _protect_small_faces(self, faces, groups):
        """Restore raw vertices on the chains of any face that simplification
        would collapse below a valid polygon.

        A sliver group bounded by two or three chains that each straighten to
        one segment ends up with a 2-vertex loop. Letting the face fall back to
        its raw boundary would stop it sharing edges with neighbours that did
        simplify, so the chains are un-simplified instead, keeping both sides
        in agreement at the cost of a few extra points on small faces."""
        self.protected = 0
        for _ in range(3):
            changed = False
            for group in groups:
                loops, ok = _fit.boundary_loops(faces, group)
                if not ok or not loops:
                    continue
                for loop in loops:
                    if len(self.simplify_loop(loop)) >= 3:
                        continue
                    for i in range(len(loop)):
                        a, b = loop[i], loop[(i + 1) % len(loop)]
                        key = (a, b) if a < b else (b, a)
                        cid = self._chain_of_edge.get(key)
                        if cid is not None and self._simplified[cid] != self._chains[cid]:
                            self._simplified[cid] = list(self._chains[cid])
                            self.protected += 1
                            changed = True
            if not changed:
                break

    def _protect_crossing_faces(self, faces, groups):
        """Restore raw vertices on the chains of any flat face whose simplified
        outline crosses itself.

        Chains are simplified one at a time; where a face is narrower than the
        tolerance, the straight line replacing one side can cut through the
        simplified chains on the other side, and the written outline crosses
        itself (Open CASCADE: a self-intersecting wire, filled wrongly on
        import). The raw chains of that face do not cross, so they are put
        back, on the chains rather than on the one face so that both neighbours
        keep the same points. A face whose raw outline already crosses is left
        alone."""
        self.crossing_protected = 0
        for _ in range(3):
            changed = False
            for gid, group in enumerate(groups):
                if gid in self._curved or len(group) < 2:
                    continue
                loops, ok = _fit.boundary_loops(faces, group)
                if not ok or not loops:
                    continue
                written = [self.simplify_loop(loop) for loop in loops]
                if (_fit.loop_crossings(written, self.verts) == 0
                        or _fit.loop_crossings(loops, self.verts) > 0):
                    continue
                for loop in loops:
                    for i in range(len(loop)):
                        a, b = loop[i], loop[(i + 1) % len(loop)]
                        key = (a, b) if a < b else (b, a)
                        cid = self._chain_of_edge.get(key)
                        if cid is not None and self._simplified[cid] != self._chains[cid]:
                            self._simplified[cid] = list(self._chains[cid])
                            self.crossing_protected += 1
                            changed = True
            if not changed:
                break

    # -- use ----------------------------------------------------------
    def simplify_loop(self, loop: Sequence[int], with_spans: bool = False):
        """Rewrite one face's raw boundary loop using the shared simplified
        chains: walking the loop edge by edge, every maximal run belonging to
        one chain is replaced by that chain's stored vertex sequence, oriented
        to the direction of travel, so both faces sharing a chain emit the same
        sequence.

        `with_spans=True` additionally returns a `{(a, b): [raw vertex ids]}`
        map giving, for every edge `(a, b)` of the output loop, the raw
        mesh-vertex run it collapsed (see `_spans_for_chain`); the writer fits
        a curve through such a run. The default returns just the vertex list."""
        n = len(loop)
        if n < 3:
            return (list(loop), {}) if with_spans else list(loop)
        cids = []
        for i in range(n):
            a = loop[i]
            b = loop[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            cids.append(self._chain_of_edge.get(key))

        # Start at a node (where the chain changes): `loop[0]` can lie in the
        # middle of a chain, and a chain is substituted whole, from one end.
        start = None
        for i in range(n):
            if cids[i] != cids[i - 1]:
                start = i
                break
        if start is None:
            # Every edge of this loop is on the same chain: the loop is one
            # closed ring (a bore rim shared with exactly one other face).
            cid = cids[0]
            if cid is None:
                return (list(loop), {}) if with_spans else list(loop)
            chain = self._chains[cid]
            simp = self._simplified[cid]
            ring = chain[:-1] if chain[0] == chain[-1] else chain
            sring = simp[:-1] if simp[0] == simp[-1] else simp
            if loop[0] not in ring:
                self.fallbacks += 1
                return (list(loop), {}) if with_spans else list(loop)
            k = ring.index(loop[0])
            forward = ring[(k + 1) % len(ring)] == loop[1]
            # The ring's start does not matter (an EDGE_LOOP is cyclic); its
            # edges must match the face on the other side.
            out = sring if forward else list(reversed(sring))
            result = out if len(out) >= 3 else list(loop)
            if not with_spans:
                return result
            return result, _spans_for_chain(chain, simp)

        order = list(range(start, n)) + list(range(start))
        out: list[int] = []
        spans: dict[tuple[int, int], list[int]] = {}
        i = 0
        while i < n:
            idx = order[i]
            a = loop[idx]
            cid = cids[idx]
            if cid is None:
                # Not a registered boundary edge (shouldn't happen for a
                # well-formed group, but never drop geometry over it).
                out.append(a)
                i += 1
                continue
            chain = self._chains[cid]
            simp = self._simplified[cid]
            length = len(chain) - 1
            if i + length > n or a not in (chain[0], chain[-1]):
                # The loop does not traverse this chain end to end from here
                # (should be impossible: chains end at nodes and the walk
                # starts at one); fall back to raw vertices for this stretch.
                self.fallbacks += 1
                out.extend(loop[order[j]] for j in range(i, n))
                break
            forward = chain[0] == a
            seq = simp if forward else list(reversed(simp))
            out.extend(seq[:-1])
            if with_spans:
                spans.update(_spans_for_chain(chain, simp))
            i += length
        result = out if len(out) >= 3 else list(loop)
        return (result, spans) if with_spans else result

    # -- diagnostics ---------------------------------------------------
    def stats(self) -> dict:
        raw = sum(len(c) - 1 for c in self._chains)
        simp = sum(len(c) - 1 for c in self._simplified)
        return {"chains": len(self._chains), "raw_edges": raw,
                "simplified_edges": simp,
                "nodes": len(self._nodes)}
