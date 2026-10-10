"""Regression tests for the STEP Part 21 parser and colour lookup.

The bug fixed here: an inline typed value such as ``NULL_STYLE(.NULL.)``
inside a set (``PRESENTATION_STYLE_ASSIGNMENT((NULL_STYLE(.NULL.)))``) is
parsed as an ``Instance(id=-1)``. ``StepFile.get`` used to call ``int()`` on
it, raising ``TypeError`` and aborting an otherwise valid import.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARSER_PATH = os.path.join(_HERE, "..", "core", "parser.py")

# parser.py is dependency-free, so load it directly: importing the ``core``
# package would pull in NumPy, which the parser-level cases do not need.
_spec = importlib.util.spec_from_file_location("step_forge_parser", _PARSER_PATH)
assert _spec is not None and _spec.loader is not None
parser = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = parser
_spec.loader.exec_module(parser)

Instance = parser.Instance
parse_string = parser.parse_string

try:
    from core.convert import styled_colours
    _HAVE_CONVERT = True
except ImportError:  # NumPy (or the rest of the package) not available
    styled_colours = None
    _HAVE_CONVERT = False


NULL_STYLE_STEP = """ISO-10303-21;
HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=PRESENTATION_STYLE_ASSIGNMENT((NULL_STYLE(.NULL.)));
#3=STYLED_ITEM('',(#2),#1);
ENDSEC;
END-ISO-10303-21;
"""

COLOUR_STEP = """ISO-10303-21;
HEADER;
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=PRESENTATION_STYLE_ASSIGNMENT((#4));
#3=STYLED_ITEM('',(#2),#1);
#4=SURFACE_STYLE_USAGE(.BOTH.,SURFACE_SIDE_STYLE('',(#5)));
#5=SURFACE_STYLE_FILL_AREA(FILL_AREA_STYLE('',(#6)));
#6=FILL_AREA_STYLE_COLOUR('',COLOUR_RGB('',0.2,0.4,0.6));
ENDSEC;
END-ISO-10303-21;
"""


class GetTests(unittest.TestCase):
    def test_inline_typed_value_is_returned(self):
        sf = parse_string(NULL_STYLE_STEP)
        psa = sf.get(2)
        inline = psa.params[0][0]
        self.assertIsInstance(inline, Instance)
        self.assertEqual(inline.name, "NULL_STYLE")
        # The regression: this used to raise TypeError.
        self.assertIs(sf.get(inline), inline)

    def test_unresolvable_ref_returns_none(self):
        sf = parse_string(NULL_STYLE_STEP)
        self.assertIsNone(sf.get("not a reference"))
        self.assertIsNone(sf.get(999999))
        self.assertIsNone(sf.get(None))
        self.assertIsNone(sf.get(parser.DERIVED))

    def test_styled_item_with_null_style_does_not_crash(self):
        sf = parse_string(NULL_STYLE_STEP)
        si = next(iter(sf.of_type("STYLED_ITEM")))
        psa_ref = si.params[1][0]
        styles = sf.get(psa_ref).params[0]
        for st_ref in styles:  # the exact loop that used to crash
            sf.get(st_ref)


@unittest.skipUnless(_HAVE_CONVERT, "core.convert requires NumPy")
class StyledColoursTests(unittest.TestCase):
    def test_null_style_yields_no_colour(self):
        assert styled_colours is not None
        sf = parse_string(NULL_STYLE_STEP)
        self.assertEqual(styled_colours(sf), {})

    def test_colour_round_trip(self):
        assert styled_colours is not None
        sf = parse_string(COLOUR_STEP)
        self.assertEqual(styled_colours(sf).get(1), (0.2, 0.4, 0.6, 1.0))


if __name__ == "__main__":
    unittest.main()
