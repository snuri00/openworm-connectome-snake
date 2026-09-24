"""
Soma positions and cell types of every neuron, read from the OpenWorm c302
NeuroML network (examples/c302_A_Full.net.nml).

Coordinates are in micrometres: y runs head (-) to tail (+), x is left/right,
z is dorsal/ventral.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NML = os.path.join(HERE, "c302", "examples", "c302_A_Full.net.nml")

_POP = re.compile(r'<population id="([^"]+)".*?</population>', re.S)
_LOC = re.compile(r'<location x="([-\d.eE]+)" y="([-\d.eE]+)" z="([-\d.eE]+)"')
_TYPE = re.compile(r'tag="type" value="([^"]+)"')


def canonical(name):
    """NeuroML uses DA9/VB11, the edge list uses DA09/VB11."""
    m = re.fullmatch(r"([A-Z]{2})(\d)", name)
    return f"{m.group(1)}0{m.group(2)}" if m else name


def load_atlas(path=DEFAULT_NML):
    """Returns {neuron name: ((x, y, z), type string)}."""
    with open(path) as f:
        text = f.read()
    atlas = {}
    for block in _POP.finditer(text):
        name = canonical(block.group(1))
        loc = _LOC.search(block.group(0))
        kind = _TYPE.search(block.group(0))
        if loc:
            atlas[name] = (tuple(float(v) for v in loc.groups()),
                           kind.group(1) if kind else "unknown")
    return atlas
