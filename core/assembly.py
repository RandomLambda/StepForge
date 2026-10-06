"""STEP assembly structure: the NEXT_ASSEMBLY_USAGE_OCCURRENCE tree and the
placement transform of each instance. If a transform chain cannot be resolved
the caller falls back to a flat list.
"""
from __future__ import annotations

import numpy as np

from . import geometry as G


def _mat_from_axis2(sf, ref):
    """4x4 homogeneous matrix from an AXIS2_PLACEMENT_3D."""
    f = G.Frame.from_axis2(sf, ref)
    m = np.eye(4)
    m[:3, 0] = f.x
    m[:3, 1] = f.y
    m[:3, 2] = f.z
    m[:3, 3] = f.o
    return m


def _item_transform(sf, idt):
    """Matrix of an ITEM_DEFINED_TRANSFORMATION: M2 @ inv(M1) of its two
    AXIS2_PLACEMENT_3D (identity if one is missing or M1 is singular)."""
    p = idt.params
    a1 = None
    a2 = None
    for x in p:
        inst = sf.get(x) if not isinstance(x, str) else None
        if inst is not None and inst.name == "AXIS2_PLACEMENT_3D":
            if a1 is None:
                a1 = x
            else:
                a2 = x
    if a1 is None or a2 is None:
        return np.eye(4)
    M1 = _mat_from_axis2(sf, a1)
    M2 = _mat_from_axis2(sf, a2)
    try:
        return M2 @ np.linalg.inv(M1)
    except np.linalg.LinAlgError:
        return np.eye(4)


def _nauo_transform(sf, nauo_id):
    """Find the transform associated with a NAUO via a
    CONTEXT_DEPENDENT_SHAPE_REPRESENTATION -> (REPRESENTATION_RELATIONSHIP +
    REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION -> ITEM_DEFINED_TRANSFORMATION)."""
    for cdsr in sf.of_type("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"):
        # params: (representation_relation, represented_product_relation=PDS->NAUO)
        rpr = cdsr.params[1] if len(cdsr.params) > 1 else None
        pds = sf.get(rpr)
        if pds is None:
            continue
        # PRODUCT_DEFINITION_SHAPE.definition may point at the NAUO
        definition = pds.params[-1] if pds.params else None
        if definition is None or int(definition) != int(nauo_id):
            continue
        rr = sf.get(cdsr.params[0])
        if rr is None:
            continue
        rec = sf.subrecord(rr, "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION") or rr
        # transformation is the last param referencing an ITEM_DEFINED_TRANSFORMATION
        for x in rec.params:
            idt = sf.get(x) if not isinstance(x, (str, list)) else None
            if idt is not None and idt.name == "ITEM_DEFINED_TRANSFORMATION":
                return _item_transform(sf, idt)
    return np.eye(4)


def build_tree(sf):
    """Return (roots, children) where children[pd_id] = list of
    (child_pd_id, 4x4 matrix, instance_name). roots are PD ids that are never a
    child. Returns None if there is no assembly usage at all."""
    nauos = sf.of_type("NEXT_ASSEMBLY_USAGE_OCCURRENCE")
    if not nauos:
        return None
    children = {}
    child_ids = set()
    for n in nauos:
        p = n.params
        # (id, name, description, relating_pd(parent), related_pd(child), ...)
        parent = int(p[3])
        child = int(p[4])
        inst_name = str(p[1]) if p[1] else ""
        mat = _nauo_transform(sf, n.id)
        children.setdefault(parent, []).append((child, mat, inst_name))
        child_ids.add(child)
    all_parents = set(children.keys())
    roots = [pid for pid in all_parents if pid not in child_ids]
    if not roots:  # fall back: any PD with children
        roots = list(all_parents)
    return roots, children
