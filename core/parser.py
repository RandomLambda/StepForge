"""ISO 10303-21 (STEP Part 21) clear-text parser.

Dependency-free. Produces a flat table of instances:
    id (int) -> Instance(name=ENTITY_NAME, params=[...])
where params is a list of python values:
    - float / int            for numbers
    - str                    for quoted strings (decoded)
    - Ref(int)               for #id references
    - Enum(str)              for .ENUM.
    - list                   for (...) aggregates
    - None                   for $  (unset)
    - Derived()              for *  (derived)
    - list[Instance]         for complex/AND records ( SUB1(..)SUB2(..) )

The parser is tolerant: unknown entities are kept verbatim so name extraction
and geometry extraction can pick out only what they understand.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


class Ref(int):
    """An entity reference (#id). Subclasses int so it can index directly."""
    __slots__ = ()
    def __repr__(self):  # pragma: no cover - debug only
        return f"#{int(self)}"


class Enum(str):
    """A STEP enumeration value written as .NAME. (without the dots)."""
    __slots__ = ()
    def __repr__(self):  # pragma: no cover
        return f".{self!s}."


class Derived:
    """The '*' derived-value marker."""
    __slots__ = ()
    _inst = None
    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst
    def __repr__(self):  # pragma: no cover
        return "*"


DERIVED = Derived()

# What a malformed record can raise while it is parsed.
_PARSE_ERRORS = (ValueError, IndexError, RecursionError)


@dataclass
class Instance:
    id: int
    name: str
    params: list[Any]
    # For complex records the top-level "name" is "" and `records` holds the
    # individual simple records.
    records: list[Instance] | None = None

    def __repr__(self):  # pragma: no cover
        return f"#{self.id}={self.name}({len(self.params)} params)"


# ---------------------------------------------------------------------------
# Tokeniser / value parser
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_comments(text: str) -> str:
    return _COMMENT_RE.sub(" ", text)


def _decode_string(raw: str) -> str:
    """Decode a Part 21 string body (without surrounding quotes): '' -> ' and
    the escape sequences for non-ASCII characters (X2, X4, X, S)."""
    s = raw.replace("''", "'")
    if "\\" not in s:
        return s

    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        # control directive
        if s[i:i + 3] == r"\X2" and s[i + 3:i + 4] == "\\":
            # \X2\ <hex quads> \X0\
            j = s.find("\\X0\\", i + 4)
            if j == -1:
                out.append(c)
                i += 1
                continue
            hexpart = s[i + 4:j]
            try:
                for k in range(0, len(hexpart), 4):
                    out.append(chr(int(hexpart[k:k + 4], 16)))
            except ValueError:
                pass
            i = j + 4
        elif s[i:i + 3] == r"\X4" and s[i + 3:i + 4] == "\\":
            j = s.find("\\X0\\", i + 4)
            if j == -1:
                out.append(c)
                i += 1
                continue
            hexpart = s[i + 4:j]
            try:
                for k in range(0, len(hexpart), 8):
                    out.append(chr(int(hexpart[k:k + 8], 16)))
            except ValueError:
                pass
            i = j + 4
        elif s[i:i + 2] == r"\X" and s[i + 3:i + 4] == "\\":
            # \X\HH  single byte (latin-1)
            try:
                out.append(chr(int(s[i + 2:i + 4], 16)))
                i += 5
            except ValueError:
                out.append(c)
                i += 1
        elif s[i:i + 2] == "\\\\":
            out.append("\\")          # \\ is an escaped reverse solidus
            i += 2
        elif s[i:i + 2] == r"\S" and s[i + 2:i + 3] == "\\":
            # \S\x : x + 128 in code page (approximate as latin-1)
            ch = s[i + 3:i + 4]
            if ch:
                out.append(chr(ord(ch) + 128))
            i += 4
        else:
            out.append(c)
            i += 1
    return "".join(out)


# A token pattern covering the building blocks of a parameter list.
_TOKEN_RE = re.compile(
    r"""
      (?P<string>   ' (?: [^'] | '' )* ' )         # 'quoted, with '' escapes'
    | (?P<ref>      \# \d+ )                        # #123
    | (?P<enum>     \. [A-Za-z0-9_]+ \. )           # .ENUM.
    | (?P<real>     [+-]? (?:\d+\.\d*|\.\d+|\d+\.) (?:E[+-]?\d+)? )  # 1.2E3
    | (?P<int>      [+-]? \d+ )                      # 123
    | (?P<lparen>   \( )
    | (?P<rparen>   \) )
    | (?P<comma>    , )
    | (?P<derived>  \* )
    | (?P<unset>    \$ )
    | (?P<keyword>  [A-Za-z_!][A-Za-z0-9_]* )       # ENTITY name
    | (?P<ws>       \s+ )
    """,
    re.VERBOSE,
)


class _Cursor:
    __slots__ = ("i", "toks")

    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t


def _tokenize(s: str):
    toks = []
    for m in _TOKEN_RE.finditer(s):
        kind = m.lastgroup
        if kind == "ws":
            continue
        toks.append((kind, m.group()))
    return toks


def _parse_value(cur: _Cursor):
    kind, val = cur.next()
    if kind == "string":
        return _decode_string(val[1:-1])
    if kind == "ref":
        return Ref(int(val[1:]))
    if kind == "enum":
        return Enum(val[1:-1])
    if kind == "real":
        return float(val)
    if kind == "int":
        return int(val)
    if kind == "unset":
        return None
    if kind == "derived":
        return DERIVED
    if kind == "lparen":
        return _parse_list(cur)
    if kind == "keyword":
        # typed value: KEYWORD( ... )  -> represent as a sub Instance(id=-1)
        nk, _ = cur.peek()
        if nk == "lparen":
            cur.next()  # consume (
            params = _parse_list(cur)
            return Instance(id=-1, name=val, params=params)
        return Enum(val)  # bare keyword (rare) treat as token
    raise ValueError(f"Unexpected token {kind}:{val}")


def _parse_list(cur: _Cursor):
    """Parse the remainder of a '(' ... ')' aggregate (the '(' already consumed)."""
    items = []
    kind, val = cur.peek()
    if kind == "rparen":
        cur.next()
        return items
    while True:
        items.append(_parse_value(cur))
        kind, val = cur.next()
        if kind == "comma":
            continue
        if kind == "rparen":
            break
        raise ValueError(f"Expected , or ) got {kind}:{val}")
    return items


def _parse_rhs(body: str) -> Instance:
    """Parse the right-hand side of '#id = <body>' into an Instance (id filled later)."""
    toks = _tokenize(body)
    cur = _Cursor(toks)
    kind, val = cur.peek()
    if kind == "lparen":
        # complex / AND record:  ( NAME1(...) NAME2(...) ... )
        cur.next()
        records = []
        while True:
            k, v = cur.peek()
            if k == "rparen":
                cur.next()
                break
            if k != "keyword":
                raise ValueError(f"Bad complex record token {k}:{v}")
            cur.next()
            # expect (
            k2, _ = cur.next()
            if k2 != "lparen":
                raise ValueError("Expected ( after subrecord name")
            params = _parse_list(cur)
            records.append(Instance(id=-1, name=v, params=params))
        # merge params for convenience
        merged: list[Any] = []
        for r in records:
            merged.extend(r.params)
        return Instance(id=-1, name="", params=merged, records=records)
    elif kind == "keyword":
        cur.next()
        k2, _ = cur.next()
        if k2 != "lparen":
            raise ValueError("Expected ( after entity name")
        params = _parse_list(cur)
        return Instance(id=-1, name=val, params=params)
    else:
        raise ValueError(f"Cannot parse rhs starting with {kind}:{val}")


# ---------------------------------------------------------------------------
# Statement splitting
# ---------------------------------------------------------------------------

def _split_statements(data: str):
    """Yield raw statements from a DATA body, splitting on ';' but not inside
    quoted strings."""
    buf = []
    in_str = False
    i = 0
    n = len(data)
    while i < n:
        c = data[i]
        if in_str:
            buf.append(c)
            if c == "'":
                if i + 1 < n and data[i + 1] == "'":
                    buf.append("'")
                    i += 2
                    continue
                in_str = False
            i += 1
            continue
        if c == "'":
            in_str = True
            buf.append(c)
            i += 1
            continue
        if c == ";":
            stmt = "".join(buf).strip()
            if stmt:
                yield stmt
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        yield tail


_HEADER_ENTITY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$", re.DOTALL)
_INSTANCE_RE = re.compile(r"^#(\d+)\s*=\s*(.*)$", re.DOTALL)


@dataclass
class StepFile:
    instances: dict[int, Instance] = field(default_factory=dict)
    header: dict[str, Instance] = field(default_factory=dict)
    schema: str = ""

    # --- lookup helpers -----------------------------------------------------
    def get(self, ref) -> Instance | None:
        # An inline typed value (`NULL_STYLE(.NULL.)`, `LENGTH_MEASURE(25.4)`)
        # is parsed as an Instance(id=-1); it is already the entity the caller
        # is after, so hand it straight back instead of looking it up.
        if isinstance(ref, Instance):
            return ref
        if ref is None or isinstance(ref, Derived):
            return None
        try:
            return self.instances.get(int(ref))
        except (TypeError, ValueError):
            # A malformed or unexpected reference value must never abort a
            # whole import: treat it as an unresolved entity.
            return None

    def of_type(self, *names: str) -> list[Instance]:
        names = set(names)
        out = []
        for inst in self.instances.values():
            if inst.name in names or (
                    inst.records and any(r.name in names for r in inst.records)):
                out.append(inst)
        return out

    def subrecord(self, inst: Instance, name: str) -> Instance | None:
        if inst.name == name:
            return inst
        if inst.records:
            for r in inst.records:
                if r.name == name:
                    return r
        return None


def parse_string(text: str) -> StepFile:
    text = _strip_comments(text)

    # Separate HEADER and DATA sections.
    sf = StepFile()
    # header
    hm = re.search(r"HEADER;(.*?)ENDSEC;", text, re.DOTALL)
    if hm:
        for stmt in _split_statements(hm.group(1)):
            m = _HEADER_ENTITY_RE.match(stmt.strip())
            if not m:
                continue
            name = m.group(1).upper()
            try:
                inst = _parse_rhs(stmt.strip())
                sf.header[name] = inst
            except _PARSE_ERRORS:
                pass

    # schema
    sm = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'", text)
    if sm:
        sf.schema = sm.group(1)

    # data (may be multiple DATA sections)
    for dm in re.finditer(r"DATA[^;]*;(.*?)ENDSEC;", text, re.DOTALL):
        for stmt in _split_statements(dm.group(1)):
            im = _INSTANCE_RE.match(stmt)
            if not im:
                continue
            sid = int(im.group(1))
            body = im.group(2)
            try:
                inst = _parse_rhs(body)
            except _PARSE_ERRORS:
                # keep an empty placeholder so references don't crash
                inst = Instance(id=sid, name="!PARSE_ERROR", params=[])
            inst.id = sid
            if inst.records:
                for r in inst.records:
                    r.id = sid
            sf.instances[sid] = inst
    return sf


def parse_file(path: str) -> StepFile:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return parse_string(fh.read())
