"""Tests for the assembly tree's NAUO transform index.

`build_tree` used to look up each NAUO's placement by scanning every
CONTEXT_DEPENDENT_SHAPE_REPRESENTATION record, i.e. O(NAUOs x CDSRs). It now
builds the whole NAUO -> transform map in one pass. These tests pin the
indexing behaviour with a lightweight stand-in for `StepFile`.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    from core import assembly as A
    _HAVE = True
except ImportError:
    _HAVE = False


class _Inst:
    def __init__(self, iid, name, params):
        self.id = iid
        self.name = name
        self.params = params
        self.records = []


class _SF:
    def __init__(self, by_id, by_type):
        self._by_id = by_id
        self._by_type = by_type

    def get(self, ref):
        if ref is None or isinstance(ref, str):
            return None
        return self._by_id.get(int(ref))

    def of_type(self, name):
        return self._by_type.get(name, [])

    def subrecord(self, inst, name):
        return inst if inst.name == name else None


def _pair(nauo_id):
    """A CDSR + PDS + RR + identity IDT chain for one NAUO."""
    rr_id, pds_id, idt_id, cdsr_id = nauo_id + 1000, nauo_id + 2000, nauo_id + 3000, nauo_id + 4000
    rr = _Inst(rr_id, "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION", [idt_id])
    idt = _Inst(idt_id, "ITEM_DEFINED_TRANSFORMATION", [])   # no AXIS2 -> identity
    pds = _Inst(pds_id, "PRODUCT_DEFINITION_SHAPE", ["", nauo_id])
    cdsr = _Inst(cdsr_id, "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION", [rr_id, pds_id])
    return cdsr, {rr_id: rr, idt_id: idt, pds_id: pds, cdsr_id: cdsr}


@unittest.skipUnless(_HAVE, "core (NumPy) not importable")
class NaoTransformIndexTests(unittest.TestCase):
    def test_index_maps_every_nauo(self):
        by_id, cdsrs = {}, []
        for nauo_id in (10, 20, 30):
            cdsr, extra = _pair(nauo_id)
            by_id.update(extra)
            cdsrs.append(cdsr)
        sf = _SF(by_id, {"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION": cdsrs})

        index = A._nauo_transforms(sf)
        self.assertEqual(sorted(index), [10, 20, 30])
        for m in index.values():
            self.assertTrue(np.allclose(m, np.eye(4)))

    def test_build_tree_assigns_indexed_transforms(self):
        by_id, cdsrs, nauos = {}, [], []
        for i, (parent, child) in enumerate([(1, 2), (2, 3)]):
            cdsr, extra = _pair(100 + i)
            by_id.update(extra)
            cdsrs.append(cdsr)
            nauos.append(_Inst(100 + i, "NEXT_ASSEMBLY_USAGE_OCCURRENCE",
                               [100 + i, f"a{i}", "", parent, child]))
        sf = _SF(by_id, {
            "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION": cdsrs,
            "NEXT_ASSEMBLY_USAGE_OCCURRENCE": nauos,
        })

        roots, children = A.build_tree(sf)
        self.assertEqual(roots, [1])
        self.assertEqual(children[1][0][0], 2)
        self.assertEqual(children[2][0][0], 3)
        self.assertTrue(np.allclose(children[1][0][1], np.eye(4)))


if __name__ == "__main__":
    unittest.main()
