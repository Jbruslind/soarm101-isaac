"""Procedural ISO metric fasteners (M3-M8) as simulation-ready USD rigid bodies.

Part types
    ``hex_bolt``     ISO 4017  hexagon head screw, thread to the head, product grade A
    ``socket_cap``   ISO 4762  hexagon socket head cap screw (SHCS)
    ``hex_nut``      ISO 4032  hexagon nut, style 1, product grade A
    ``washer``       ISO 7089  plain washer, normal series, product grade A

Everything except :func:`write_usd` is plain Python (no numpy, no ``pxr``, no Isaac) so the
dimension tables, meshes and mass maths can be unit-tested anywhere. ``pxr`` is imported lazily
inside the USD writer, like Isaac Lab's ``contrib/keyboard/keyboards/keyboard_usd.py``.

Conventions (all lengths in metres in the USD, millimetres in the tables)
    * z is the fastener axis. For screws the bearing face of the head is z = 0, the head is in
      z > 0 and the shank runs down to z = -length. Nuts and washers sit on z = 0.
    * The default prim ``/Fastener`` carries ``RigidBodyAPI`` + ``MassAPI``, so the file loads with
      ``sim_utils.UsdFileCfg`` as a ``RigidObject`` and can be one entry of a
      ``MultiAssetSpawnerCfg`` (see :func:`fastener_spawn_cfg`).
    * ``/Fastener/visual`` is one closed mesh (no threads; flats, chamfers, washer face and the
      hex socket are modelled). ``/Fastener/collisions/*`` are convex hulls (purpose ``guide``):
      shank = 16-gon prism, head = hexagonal prism (ISO 4017) or 16-gon prism (ISO 4762), nut =
      6 convex sectors, washer = 16-gon disc without the hole (M3: 4 boxes, see collision_shapes);
      ``physxCollision:contactOffset`` = 1 mm, ``restOffset`` = 0.

Mass
    Mass, centre of mass and inertia come from a closed "mass mesh" (same profile as the visual
    mesh, 360 angular samples) at steel density 7850 kg/m^3. The threaded part of a shank or a
    nut bore is taken at the pitch diameter d2 instead of the major diameter d: the ISO basic
    thread profile is symmetric about the pitch line, so a thread holds as much steel as a
    plain d2 cylinder. The per-millimetre mass increments of the DIN 933 / DIN 912 weight
    tables confirm this (e.g. M8: 0.320 g/mm table vs 0.3186 g/mm for a d2 = 7.188 mm rod).

Grasp
    ``grasp_width_mm`` is the width the SO-ARM101 jaws close on: the shank major diameter for
    screws (gripped across the shank), across-flats for nuts, the outside diameter for washers.
    ``jaw_contact_angle_rad = 2*asin(w / 0.186)`` (w in metres) is the jaw opening angle at that
    width, used by the tutorials' jaw-fit maths.

Output
    ``isaac_envs/assets/usd/fasteners/<name>.usda`` + ``manifest.json`` (committed; ~2 MB of text).
    It lives under isaac_envs/ because docker-compose bind-mounts that directory into the Isaac Sim
    container (``/isaac_envs/assets/usd/fasteners``). Names: ``hex_bolt_M6x20``,
    ``socket_cap_M8x40``, ``hex_nut_M6``, ``washer_M6``.

Usage in Isaac Lab (with /isaac_envs on sys.path, like the env scripts)::

    from assets.fasteners import fastener_spawn_cfg
    bolt = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Bolt", spawn=fastener_spawn_cfg("hex_bolt_M6x20"))
    mix = sim_utils.MultiAssetSpawnerCfg(assets_cfg=[fastener_spawn_cfg(n) for n in names])

CLI (needs pxr for writing, e.g. /isaac-sim/python.sh)
    ``python isaac_envs/assets/fasteners.py``             write the full set + manifest.json
    ``python isaac_envs/assets/fasteners.py --only hex_bolt_M6x20``
    ``python isaac_envs/assets/fasteners.py --list``      print the table, no pxr needed
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# Standard tables (millimetres). Nominal values only: where the standard gives "nom = max"
# (s, dk, d2, d1 of washers) that value is used; for heights the nominal k / m / h is used, and
# where a standard only gives limits (dw min, c max, t min) the cited limit is used.
# ---------------------------------------------------------------------------------------------

STEEL_DENSITY = 7850.0  # kg/m^3
JAW_RADIUS_M = 0.186 / 2  # jaw_contact_angle = 2 asin(w / (2 * 0.093))

# Metric coarse thread pitch P, ISO 261 / ISO 262.
PITCH = {"M3": 0.5, "M4": 0.7, "M5": 0.8, "M6": 1.0, "M8": 1.25}
# Nominal (major) diameter d.
NOMINAL_D = {"M3": 3.0, "M4": 4.0, "M5": 5.0, "M6": 6.0, "M8": 8.0}
SIZES = tuple(PITCH)


def pitch_diameter(size: str) -> float:
    """Basic pitch diameter d2 = d - 0.649519 P (ISO 724)."""
    return NOMINAL_D[size] - 0.649519 * PITCH[size]


def minor_diameter_internal(size: str) -> float:
    """Basic minor diameter of the internal thread D1 = d - 1.082532 P (ISO 724)."""
    return NOMINAL_D[size] - 1.082532 * PITCH[size]


def minor_diameter_external(size: str) -> float:
    """Basic minor diameter of the external thread d3 = d - 1.226869 P (ISO 724)."""
    return NOMINAL_D[size] - 1.226869 * PITCH[size]


# ISO 4017:2014 (= ISO 4014 head), table 1, product grade A:
#   s  width across flats, nom = max
#   k  head height, nom
#   dw bearing (washer-face) diameter, min
#   c  washer-face height, max
ISO4017 = {
    "M3": {"s": 5.5, "k": 2.0, "dw": 4.57, "c": 0.40},
    "M4": {"s": 7.0, "k": 2.8, "dw": 5.88, "c": 0.40},
    "M5": {"s": 8.0, "k": 3.5, "dw": 6.88, "c": 0.50},
    "M6": {"s": 10.0, "k": 4.0, "dw": 8.88, "c": 0.50},
    "M8": {"s": 13.0, "k": 5.3, "dw": 11.63, "c": 0.60},
}
# Preferred lengths l (ISO 4017 length ranges: M3 6-30, M4 8-40, M5 10-50, M6 12-60, M8 16-80).
ISO4017_LENGTHS = {
    "M3": (8, 10, 12, 16, 20),
    "M4": (8, 10, 12, 16, 20, 25),
    "M5": (10, 12, 16, 20, 25, 30),
    "M6": (12, 16, 20, 25, 30, 40),
    "M8": (16, 20, 25, 30, 40),
}

# ISO 4762:2004, table 1:
#   dk head diameter (plain head), max = nom
#   k  head height, nom = max (= d)
#   s  hexagon socket size, nom
#   t  socket depth, min
#   b  thread length, ref (2d + 12)
#   full_to  longest l that is threaded to the head (lengths above the stepped line in table 1);
#            longer screws get a plain shank of length l - b at diameter d.
ISO4762 = {
    "M3": {"dk": 5.5, "k": 3.0, "s": 2.5, "t": 1.3, "b": 18.0, "full_to": 20},
    "M4": {"dk": 7.0, "k": 4.0, "s": 3.0, "t": 2.0, "b": 20.0, "full_to": 25},
    "M5": {"dk": 8.5, "k": 5.0, "s": 4.0, "t": 2.5, "b": 22.0, "full_to": 25},
    "M6": {"dk": 10.0, "k": 6.0, "s": 5.0, "t": 3.0, "b": 24.0, "full_to": 30},
    "M8": {"dk": 13.0, "k": 8.0, "s": 6.0, "t": 4.0, "b": 28.0, "full_to": 35},
}
# ISO 4762 length ranges: M3 5-30, M4 6-40, M5 8-50, M6 10-60, M8 12-80.
ISO4762_LENGTHS = {
    "M3": (6, 8, 10, 12, 16, 20),
    "M4": (8, 10, 12, 16, 20, 25),
    "M5": (10, 12, 16, 20, 25, 30),
    "M6": (10, 12, 16, 20, 25, 30, 40),
    "M8": (12, 16, 20, 25, 30, 40),
}

# ISO 4032:2012, table 1, product grade A:  s nom = max, m max (nominal height), dw min.
ISO4032 = {
    "M3": {"s": 5.5, "m": 2.4, "dw": 4.6},
    "M4": {"s": 7.0, "m": 3.2, "dw": 5.9},
    "M5": {"s": 8.0, "m": 4.7, "dw": 6.9},
    "M6": {"s": 10.0, "m": 5.2, "dw": 8.9},
    "M8": {"s": 13.0, "m": 6.8, "dw": 11.6},
}

# ISO 7089:2000, table 1:  d1 nom = min (hole), d2 nom = max (outside), h nom (thickness).
ISO7089 = {
    "M3": {"d1": 3.2, "d2": 7.0, "h": 0.5},
    "M4": {"d1": 4.3, "d2": 9.0, "h": 0.8},
    "M5": {"d1": 5.3, "d2": 10.0, "h": 1.0},
    "M6": {"d1": 6.4, "d2": 12.0, "h": 1.6},
    "M8": {"d1": 8.4, "d2": 16.0, "h": 1.6},
}

# Reference masses, g per piece (= kg / 1000 pcs, steel 7.85 kg/dm^3) from the DIN weight tables
# of the withdrawn predecessors, whose M3-M8 dimensions equal the ISO parts above:
# DIN 933 -> ISO 4017, DIN 912 -> ISO 4762, DIN 934 -> ISO 4032 (same m only for M3/M4/M6;
# DIN 934 M5/M8 nuts are lower, 4.0/6.5 mm, so they are not used), DIN 125-A -> ISO 7089.
REFERENCE_MASS_G = {
    "hex_bolt": {
        "M3": {10: 0.75, 12: 0.84, 16: 1.00, 20: 1.18},
        "M4": {10: 1.64, 12: 1.80, 16: 2.10, 20: 2.41, 25: 2.80},
        "M5": {10: 2.63, 12: 2.87, 16: 3.37, 20: 3.87, 25: 4.45, 30: 5.11},
        "M6": {12: 4.42, 16: 5.11, 20: 5.80, 25: 6.65, 30: 7.51, 40: 9.23},
        "M8": {16: 11.1, 20: 12.3, 25: 13.9, 30: 15.5, 40: 18.7},
    },
    "socket_cap": {
        "M3": {8: 0.80, 10: 0.88, 12: 0.96, 16: 1.16, 20: 1.36},
        "M4": {8: 1.65, 10: 1.80, 12: 1.95, 16: 2.25, 20: 2.65, 25: 3.15},
        "M5": {10: 2.70, 12: 2.95, 16: 3.45, 20: 4.01, 25: 4.78, 30: 5.55},
        "M6": {10: 4.70, 12: 5.07, 16: 5.75, 20: 6.53, 25: 7.59, 30: 8.30, 40: 11.0},
        "M8": {12: 10.9, 16: 12.1, 20: 13.4, 25: 15.0, 30: 16.9, 40: 20.9},
    },
    "hex_nut": {"M3": {0: 0.384}, "M4": {0: 0.81}, "M6": {0: 2.50}},
    "washer": {"M3": {0: 0.119}, "M4": {0: 0.308}, "M5": {0: 0.443}, "M6": {0: 1.02}, "M8": {0: 1.83}},
}
REFERENCE_SOURCE = {
    "hex_bolt": "DIN 933 weight table (kg/1000 pcs)",
    "socket_cap": "DIN 912 weight table (kg/1000 pcs)",
    "hex_nut": "DIN 934 weight table (kg/1000 pcs)",
    "washer": "DIN 125-A weight table (kg/1000 pcs)",
}

STANDARD = {"hex_bolt": "ISO 4017", "socket_cap": "ISO 4762", "hex_nut": "ISO 4032", "washer": "ISO 7089"}

# Modelling choices that the standards leave open.
HEX_TOP_CHAMFER_DEG = 30.0  # ISO 4017 / ISO 4032 show a 15-30 deg chamfer on the hex corners
HEX_TOP_CHAMFER_DIA = 0.95  # chamfer circle on hex faces, fraction of s (never below dw)
SHCS_TOP_CHAMFER = 0.06  # top edge chamfer of the cylindrical head, fraction of dk
SOCKET_CONE_DEG = 118.0  # drill-point bottom of the hex socket
VISUAL_SEGMENTS = 48  # angular samples of the visual mesh (multiple of 12 keeps hex corners exact)
MASS_SEGMENTS = 360  # angular samples of the mass mesh (polygon area error < 0.01 %)


# ---------------------------------------------------------------------------------------------
# Part specification
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FastenerSpec:
    """One fastener: type ('hex_bolt' | 'socket_cap' | 'hex_nut' | 'washer'), size ('M6'), length l in mm."""

    kind: str
    size: str
    length: float = 0.0

    def __post_init__(self):
        if self.kind not in STANDARD:
            raise ValueError(f"unknown fastener type {self.kind!r}")
        if self.size not in PITCH:
            raise ValueError(f"unsupported size {self.size!r}; expected one of {SIZES}")
        if self.is_screw and self.length <= 0:
            raise ValueError(f"{self.kind} needs a length")

    @property
    def is_screw(self) -> bool:
        return self.kind in ("hex_bolt", "socket_cap")

    @property
    def name(self) -> str:
        if self.is_screw:
            return f"{self.kind}_{self.size}x{self.length:g}"
        return f"{self.kind}_{self.size}"

    @property
    def designation(self) -> str:
        if self.is_screw:
            return f"{STANDARD[self.kind]} {self.size}x{self.length:g}"
        return f"{STANDARD[self.kind]} {self.size}"

    @property
    def dims(self) -> dict:
        table = {"hex_bolt": ISO4017, "socket_cap": ISO4762, "hex_nut": ISO4032, "washer": ISO7089}[self.kind]
        return dict(table[self.size])

    @property
    def d(self) -> float:
        return NOMINAL_D[self.size]

    @property
    def threaded_length(self) -> float:
        """Length of the threaded part of the shank (mm), counted from the tip."""
        if self.kind == "hex_bolt":
            return self.length  # ISO 4017: thread to the head
        if self.kind == "socket_cap":
            dims = self.dims
            return self.length if self.length <= dims["full_to"] else dims["b"]
        return 0.0

    @property
    def head_width(self) -> float:
        """Across-flats (hex) or head diameter (SHCS) or outside diameter (washer), mm."""
        dims = self.dims
        return {"hex_bolt": dims.get("s"), "socket_cap": dims.get("dk"), "hex_nut": dims.get("s"),
                "washer": dims.get("d2")}[self.kind]

    @property
    def head_height(self) -> float:
        dims = self.dims
        return {"hex_bolt": dims.get("k"), "socket_cap": dims.get("k"), "hex_nut": dims.get("m"),
                "washer": dims.get("h")}[self.kind]

    @property
    def grasp_width(self) -> float:
        """Width the jaws close on (mm): shank d for screws, across flats for nuts, outside dia for washers."""
        if self.is_screw:
            return self.d
        return self.head_width


def default_specs() -> list[FastenerSpec]:
    """The generated set: all ISO 4017 / ISO 4762 lengths above, plus one nut and one washer per size."""
    specs = [FastenerSpec("hex_bolt", "M6", 20)]  # Part 1 example first
    for size in SIZES:
        specs += [FastenerSpec("hex_bolt", size, l) for l in ISO4017_LENGTHS[size]]
        specs += [FastenerSpec("socket_cap", size, l) for l in ISO4762_LENGTHS[size]]
        specs += [FastenerSpec("hex_nut", size), FastenerSpec("washer", size)]
    seen, out = set(), []
    for spec in specs:
        if spec.name not in seen:
            seen.add(spec.name)
            out.append(spec)
    return out


def spec_from_name(name: str) -> FastenerSpec:
    """Inverse of :attr:`FastenerSpec.name`, e.g. 'hex_bolt_M6x20' or 'washer_M8'."""
    kind, _, rest = name.rpartition("_")
    if "x" in rest:
        size, length = rest.split("x")
        return FastenerSpec(kind, size, float(length))
    return FastenerSpec(kind, rest)


# ---------------------------------------------------------------------------------------------
# Lofted profile meshes
# ---------------------------------------------------------------------------------------------


def hex_radius(across_flats: float, theta: float) -> float:
    """Distance from the axis to a regular hexagon's boundary at polar angle theta (flat normals at 0, 60, ...)."""
    sector = math.pi / 3.0
    phi = (theta + sector / 2.0) % sector - sector / 2.0
    return 0.5 * across_flats / math.cos(phi)


def _profile(spec: FastenerSpec, theta: float, mass_model: bool) -> tuple[list[tuple[float, float]], bool]:
    """Closed r-z profile of the part at polar angle theta, in mm.

    Returns ``(points, on_axis)``. When ``on_axis`` is True the loop starts and ends on the axis
    (r = 0) and the closing segment runs along the axis (solid part); otherwise the loop is an
    annulus cross-section (nut, washer) closed by its last-to-first segment.
    """
    d = spec.d
    dims = spec.dims
    tan30 = math.tan(math.radians(HEX_TOP_CHAMFER_DEG))
    r_major = d / 2.0
    r_d2 = pitch_diameter(spec.size) / 2.0
    r_thread = r_d2 if mass_model else r_major

    if spec.is_screw:
        l = spec.length
        # ISO 4753 chamfered end: 45 deg down to the thread minor diameter.
        r_tip = minor_diameter_external(spec.size) / 2.0
        tip = r_thread - r_tip
        pts = [(0.0, -l), (r_tip, -l), (r_thread, -l + tip)]
        lt = spec.threaded_length
        if lt < l:  # plain shank (diameter d) above the thread
            pts += [(r_thread, -l + lt), (r_major, -l + lt), (r_major, 0.0)]
        else:
            pts += [(r_thread, 0.0)]

        if spec.kind == "hex_bolt":
            s, k, dw, c = dims["s"], dims["k"], dims["dw"], dims["c"]
            big_r = hex_radius(s, theta)
            r_w = min(big_r, dw / 2.0)
            r_top = max(HEX_TOP_CHAMFER_DIA * s, dw) / 2.0
            z_ch = max(c, k - (big_r - r_top) * tan30)
            pts += [(r_w, 0.0), (r_w, c), (big_r, c), (big_r, z_ch), (r_top, k), (0.0, k)]
        else:
            dk, k, s, t = dims["dk"], dims["k"], dims["s"], dims["t"]
            ch = SHCS_TOP_CHAMFER * dk
            r_s = hex_radius(s, theta)
            e_s = s / math.cos(math.pi / 6.0)
            cone = (e_s / 2.0) / math.tan(math.radians(SOCKET_CONE_DEG / 2.0))
            pts += [(dk / 2.0, 0.0), (dk / 2.0, k - ch), (dk / 2.0 - ch, k), (r_s, k), (r_s, k - t),
                    (0.0, k - t - cone)]
        return pts, True

    if spec.kind == "hex_nut":
        s, m, dw = dims["s"], dims["m"], dims["dw"]
        big_r = hex_radius(s, theta)
        r_w = max(HEX_TOP_CHAMFER_DIA * s, dw) / 2.0  # chamfer circle on both faces
        z1 = min((big_r - r_w) * tan30, 0.45 * m)
        r_bore = r_d2 if mass_model else minor_diameter_internal(spec.size) / 2.0
        return [(r_bore, 0.0), (r_w, 0.0), (big_r, z1), (big_r, m - z1), (r_w, m), (r_bore, m)], False

    # washer
    d1, d2, h = dims["d1"], dims["d2"], dims["h"]
    return [(d1 / 2.0, 0.0), (d2 / 2.0, 0.0), (d2 / 2.0, h), (d1 / 2.0, h)], False


@dataclass
class TriMesh:
    """Closed polygon mesh. ``faces`` are lists of vertex indices, counter-clockwise seen from outside."""

    points: list[tuple[float, float, float]]
    faces: list[list[int]]

    def triangles(self):
        for f in self.faces:
            for i in range(1, len(f) - 1):
                yield self.points[f[0]], self.points[f[i]], self.points[f[i + 1]]

    def scaled(self, factor: float) -> "TriMesh":
        return TriMesh([(x * factor, y * factor, z * factor) for x, y, z in self.points], [list(f) for f in self.faces])


def loft_mesh(spec: FastenerSpec, segments: int = VISUAL_SEGMENTS, mass_model: bool = False) -> TriMesh:
    """Sweep the per-angle profile around z into one closed mesh (mm). Coincident vertices are merged."""
    if segments % 12:
        raise ValueError("segments must be a multiple of 12 so hexagon corners are sampled exactly")
    index: dict[tuple[float, float, float], int] = {}
    points: list[tuple[float, float, float]] = []

    def vid(p):
        key = (round(p[0], 9) + 0.0, round(p[1], 9) + 0.0, round(p[2], 9) + 0.0)
        if key not in index:
            index[key] = len(points)
            points.append(key)
        return index[key]

    grid = []
    on_axis = True
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        prof, on_axis = _profile(spec, theta, mass_model)
        ct, st = math.cos(theta), math.sin(theta)
        grid.append([vid((r * ct, r * st, z)) for r, z in prof])

    n_prof = len(grid[0])
    edges = range(n_prof - 1) if on_axis else range(n_prof)
    faces = []
    for i in range(segments):
        a, b = grid[i], grid[(i + 1) % segments]
        for j in edges:
            jn = (j + 1) % n_prof
            quad = [a[j], b[j], b[jn], a[jn]]
            face = []
            for v in quad:
                if v not in face:
                    face.append(v)
            if len(face) >= 3:
                faces.append(face)
    mesh = TriMesh(points, faces)
    if mass_properties(mesh, 1.0)["volume"] < 0:
        mesh.faces = [f[::-1] for f in mesh.faces]
    return mesh


# ---------------------------------------------------------------------------------------------
# Mass properties
# ---------------------------------------------------------------------------------------------


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def mass_properties(mesh: TriMesh, density: float) -> dict:
    """Volume, mass, centre of mass and inertia tensor (about the CoM) of a closed mesh.

    Sum of signed tetrahedra (origin, a, b, c). For one tetrahedron with edge vectors a, b, c the
    second moment is  int x x^T dV = det/120 * (a a^T + b b^T + c c^T + (a+b+c)(a+b+c)^T).
    Units follow the inputs (mesh in m and density in kg/m^3 -> kg, kg m^2).
    """
    vol = 0.0
    first = [0.0, 0.0, 0.0]
    second = [[0.0] * 3 for _ in range(3)]
    for a, b, c in mesh.triangles():
        det = _dot(a, _cross(b, c))
        vol += det / 6.0
        sm = (a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2])
        for k in range(3):
            first[k] += det / 24.0 * sm[k]
        for r in range(3):
            for q in range(3):
                second[r][q] += det / 120.0 * (a[r] * a[q] + b[r] * b[q] + c[r] * c[q] + sm[r] * sm[q])
    if abs(vol) < 1e-30:
        raise ValueError("degenerate mesh")
    com = tuple(f / vol for f in first)
    # second moments about the CoM
    cov = [[second[r][q] - vol * com[r] * com[q] for q in range(3)] for r in range(3)]
    trace = cov[0][0] + cov[1][1] + cov[2][2]
    inertia = [[density * ((trace if r == q else 0.0) - cov[r][q]) for q in range(3)] for r in range(3)]
    return {"volume": vol, "mass": density * vol, "com": com, "inertia": inertia}


def fastener_mass_properties(spec: FastenerSpec) -> dict:
    """Mass properties in SI units (kg, m, kg m^2) from the 360-segment mass mesh at steel density."""
    mesh = loft_mesh(spec, MASS_SEGMENTS, mass_model=True).scaled(1e-3)
    return mass_properties(mesh, STEEL_DENSITY)


def jaw_contact_angle(width_mm: float) -> float:
    """Jaw opening angle (rad) at grasp width w: 2 asin(w / 0.186), w in metres."""
    return 2.0 * math.asin((width_mm * 1e-3) / (2.0 * JAW_RADIUS_M))


def reference_mass_g(spec: FastenerSpec) -> float | None:
    table = REFERENCE_MASS_G[spec.kind].get(spec.size, {})
    return table.get(int(spec.length) if spec.is_screw else 0)


def _clean(v: float, nd: int) -> float:
    """Round and drop signed zeros / symmetry noise."""
    v = round(v, nd)
    return 0.0 if v == 0 else v


def manifest_entry(spec: FastenerSpec, usd_file: str | None = None) -> dict:
    props = fastener_mass_properties(spec)
    mass_g = props["mass"] * 1e3
    ref = reference_mass_g(spec)
    entry = {
        "name": spec.name,
        "type": spec.kind,
        "standard": STANDARD[spec.kind],
        "designation": spec.designation,
        "size": spec.size,
        "length_mm": float(spec.length) if spec.is_screw else None,
        "pitch_mm": PITCH[spec.size],
        "head_width_mm": spec.head_width,
        "head_height_mm": spec.head_height,
        "mass_g": round(mass_g, 4),
        "ref_mass_g": ref,
        "ref_mass_err_pct": round(100.0 * (mass_g - ref) / ref, 2) if ref else None,
        "grasp_width_mm": spec.grasp_width,
        "jaw_contact_angle_rad": round(jaw_contact_angle(spec.grasp_width), 5),
        "com_m": [_clean(v, 7) for v in props["com"]],
        "usd": usd_file or f"{spec.name}.usda",
    }
    return entry


# ---------------------------------------------------------------------------------------------
# Collision shapes
# ---------------------------------------------------------------------------------------------

# Round colliders (shank, SHCS head): 0 = UsdGeom.Cylinder primitives, N = convex N-gon prisms with
# their corners on the nominal circle. 16 is the default: in the Isaac Sim 6.2 drop test cylinder
# primitives let a lying SHCS creep (rolls ~0.3 mm/s, never settles) and in a larger scene fell
# through the ground box; 12-gons rocked on their corners; 16- and 24-gons settled in < 1 s.
# Known PhysX artefact: where a lying SHCS rests on the edge of its round head it sinks 0.2-0.5 mm
# into the ground (flat-face contacts, e.g. hex heads, nuts, washers, are exact to a few um);
# adding the visual chamfers to the colliders made this worse, not better.
ROUND_COLLIDER_SEGMENTS = 16
# physxConvexHullCollision:minThickness (schema default 1 mm) is authored at 0.1 mm for the 0.5 mm
# M3 washer; in the 6.2 drop test it made no difference either way (see the washer collider below).
CONVEX_MIN_THICKNESS = 1e-4  # m
THIN_HULL_LIMIT_MM = 0.8  # thinner washers get box colliders, see collision_shapes()


def _prism(polygon: list[tuple[float, float]], z0: float, z1: float) -> list[tuple[float, float, float]]:
    return [(x, y, z0) for x, y in polygon] + [(x, y, z1) for x, y in polygon]


def _hexagon(across_flats: float) -> list[tuple[float, float]]:
    r = across_flats / math.sqrt(3.0)  # corner radius
    return [(r * math.cos(math.radians(30 + 60 * i)), r * math.sin(math.radians(30 + 60 * i))) for i in range(6)]


def _round(name: str, radius: float, z0: float, z1: float, segments: int) -> tuple:
    """A round collider: UsdGeom.Cylinder primitive (segments == 0) or a convex N-gon prism (corners on the circle)."""
    if segments <= 0:
        return (name, "cylinder", radius, z1 - z0, 0.5 * (z0 + z1))
    poly = [(radius * math.cos(2 * math.pi * i / segments), radius * math.sin(2 * math.pi * i / segments))
            for i in range(segments)]
    return (name, "convex", _prism(poly, z0, z1))


def collision_shapes(spec: FastenerSpec, round_segments: int | None = None) -> list[tuple]:
    """Collision shapes in mm: ("name", "convex", points), ("name", "cylinder", radius, height, z_center)
    or ("name", "box", size_xyz, center_xyz, yaw_deg)."""
    n = ROUND_COLLIDER_SEGMENTS if round_segments is None else round_segments
    dims = spec.dims
    if spec.is_screw:
        l = spec.length
        shapes = [_round("shank", spec.d / 2.0, -l, 0.0, n)]
        if spec.kind == "hex_bolt":
            shapes.append(("head", "convex", _prism(_hexagon(dims["s"]), 0.0, dims["k"])))
        else:
            shapes.append(_round("head", dims["dk"] / 2.0, 0.0, dims["k"], n))
        return shapes
    if spec.kind == "hex_nut":
        # six convex sectors between the outer hexagon and a hexagonal hole whose flats touch d/2
        outer, inner = _hexagon(dims["s"]), _hexagon(spec.d)
        return [(f"sector_{i}", "convex", _prism([inner[i], outer[i], outer[(i + 1) % 6], inner[(i + 1) % 6]],
                                                0.0, dims["m"])) for i in range(6)]
    # washer: one solid collider, the hole is not modelled. PhysX 5 (Isaac Sim 6.2) mis-cooks convex
    # hulls thinner than ~0.5 mm (annulus sectors of the M3 washer came out 1.2 mm thick; a 16-gon M3
    # disc rested tilted 8 deg on its centre, i.e. the rim vertices were welded to the mid-plane), with
    # or without physxConvexHullCollision:minThickness. Washers of 0.8 mm and up cook exactly as a
    # 16-gon disc; thinner ones use four box primitives rotated by 45 deg, whose union is a regular
    # octagon with its corners on d2 (boxes are analytic, nothing is cooked).
    r_out = dims["d2"] / 2.0
    if dims["h"] >= THIN_HULL_LIMIT_MM or n <= 0:
        return [_round("disc", r_out, 0.0, dims["h"], n)]
    a = r_out * math.cos(math.pi / 8)
    b = r_out * math.sin(math.pi / 8)
    return [(f"box_{i}", "box", (2 * a, 2 * b, dims["h"]), (0.0, 0.0, dims["h"] / 2.0), 45.0 * i) for i in range(4)]


def collision_support_points(spec: FastenerSpec, n: int = 32,
                             round_segments: int | None = None) -> list[tuple[float, float, float]]:
    """Points (m) on the collision geometry's outer boundary, e.g. to check ground penetration."""
    pts = []
    for shape in collision_shapes(spec, round_segments):
        if shape[1] == "convex":
            pts += shape[2]
        elif shape[1] == "box":
            _, _, (sx, sy, sz), (cx, cy, cz), yaw = shape
            c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            for ex in (-0.5, 0.5):
                for ey in (-0.5, 0.5):
                    for ez in (-0.5, 0.5):
                        x, y = ex * sx, ey * sy
                        pts.append((cx + c * x - s_ * y, cy + s_ * x + c * y, cz + ez * sz))
        else:
            _, _, r, h, zc = shape
            for i in range(n):
                a = 2 * math.pi * i / n
                for z in (zc - h / 2, zc + h / 2):
                    pts.append((r * math.cos(a), r * math.sin(a), z))
    return [(x * 1e-3, y * 1e-3, z * 1e-3) for x, y, z in pts]


# ---------------------------------------------------------------------------------------------
# USD writer (pxr imported lazily)
# ---------------------------------------------------------------------------------------------

CONTACT_OFFSET = 0.001  # m
REST_OFFSET = 0.0
STATIC_FRICTION = 0.5  # steel on dry steel/plastic, order of magnitude
DYNAMIC_FRICTION = 0.4
LOOKS = {
    # zinc-plated steel (hex bolts, nuts, washers) and black-oxide steel (SHCS)
    "zinc": {"color": (0.72, 0.73, 0.75), "metallic": 1.0, "roughness": 0.35},
    "black_oxide": {"color": (0.06, 0.06, 0.065), "metallic": 0.9, "roughness": 0.45},
}


def _round_pts(points, factor=1e-3, nd=7):
    return [(round(x * factor, nd), round(y * factor, nd), round(z * factor, nd)) for x, y, z in points]


def write_usd(spec: FastenerSpec, path: str | Path, round_segments: int | None = None) -> dict:
    """Write one fastener as a .usda rigid body. Returns its manifest entry."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetMetadata("comment", f"{spec.designation}, generated by isaac_envs/assets/fasteners.py; do not edit")

    root = UsdGeom.Xform.Define(stage, "/Fastener").GetPrim()
    stage.SetDefaultPrim(root)
    Usd.ModelAPI(root).SetKind("component")
    entry = manifest_entry(spec, path.name)
    for key, val, typ in (
        ("fastener:type", spec.kind, Sdf.ValueTypeNames.String),
        ("fastener:standard", STANDARD[spec.kind], Sdf.ValueTypeNames.String),
        ("fastener:size", spec.size, Sdf.ValueTypeNames.String),
        ("fastener:lengthMm", float(spec.length), Sdf.ValueTypeNames.Float),
        ("fastener:graspWidthMm", float(spec.grasp_width), Sdf.ValueTypeNames.Float),
    ):
        root.CreateAttribute(key, typ, custom=True).Set(val)

    # --- rigid body + mass (from the mass mesh) ---
    UsdPhysics.RigidBodyAPI.Apply(root)
    root.AddAppliedSchema("PhysxRigidBodyAPI")
    root.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(1.0)
    root.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
    root.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(1)
    props = fastener_mass_properties(spec)
    inertia = props["inertia"]
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(props["mass"]))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[_clean(v, 9) for v in props["com"]]))
    # Axisymmetric (hexagonal symmetry): the principal axes are the body axes.
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(inertia[i][i]) for i in range(3)]))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # --- looks ---
    look = "black_oxide" if spec.kind == "socket_cap" else "zinc"
    UsdGeom.Scope.Define(stage, "/Fastener/Looks")
    mat = UsdShade.Material.Define(stage, f"/Fastener/Looks/{look}")
    shader = UsdShade.Shader.Define(stage, f"/Fastener/Looks/{look}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*LOOKS[look]["color"]))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(LOOKS[look]["metallic"])
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(LOOKS[look]["roughness"])
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    phys_mat = UsdShade.Material.Define(stage, "/Fastener/Looks/steel_physics")
    pm = UsdPhysics.MaterialAPI.Apply(phys_mat.GetPrim())
    pm.CreateStaticFrictionAttr(STATIC_FRICTION)
    pm.CreateDynamicFrictionAttr(DYNAMIC_FRICTION)
    pm.CreateRestitutionAttr(0.0)
    pm.CreateDensityAttr(STEEL_DENSITY)

    # --- visual mesh ---
    vis_mesh = loft_mesh(spec, VISUAL_SEGMENTS)
    vis = UsdGeom.Mesh.Define(stage, "/Fastener/visual")
    vis.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in _round_pts(vis_mesh.points)]))
    vis.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in vis_mesh.faces]))
    vis.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in vis_mesh.faces for i in f]))
    vis.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    vis.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    vis.CreateDoubleSidedAttr(False)
    lo = [min(p[k] for p in vis_mesh.points) * 1e-3 for k in range(3)]
    hi = [max(p[k] for p in vis_mesh.points) * 1e-3 for k in range(3)]
    vis.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
    vis.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*LOOKS[look]["color"])]))
    UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(mat)

    # --- colliders ---
    UsdGeom.Scope.Define(stage, "/Fastener/collisions")
    for shape in collision_shapes(spec, round_segments):
        cpath = f"/Fastener/collisions/{shape[0]}"
        if shape[1] == "box":
            _, _, size, center, yaw = shape
            cube = UsdGeom.Cube.Define(stage, cpath)
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*[c * 1e-3 for c in center]))
            cube.AddRotateZOp().Set(float(yaw))
            cube.AddScaleOp().Set(Gf.Vec3f(*[v * 1e-3 for v in size]))
            gprim = cube
        elif shape[1] == "cylinder":
            _, _, radius, height, zc = shape
            cyl = UsdGeom.Cylinder.Define(stage, cpath)
            cyl.CreateAxisAttr(UsdGeom.Tokens.z)
            cyl.CreateRadiusAttr(radius * 1e-3)
            cyl.CreateHeightAttr(height * 1e-3)
            cyl.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-radius * 1e-3, -radius * 1e-3, -height * 5e-4),
                                                Gf.Vec3f(radius * 1e-3, radius * 1e-3, height * 5e-4)]))
            cyl.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, zc * 1e-3))
            gprim = cyl
        else:
            pts = _round_pts(shape[2])
            mesh = UsdGeom.Mesh.Define(stage, cpath)
            mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
            # Hull is computed by PhysX (approximation convexHull); faces only for display/extent.
            n = len(pts) // 2
            faces = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
            faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in faces]))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*[min(p[k] for p in pts) for k in range(3)]),
                                                 Gf.Vec3f(*[max(p[k] for p in pts) for k in range(3)])]))
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            mesh.GetPrim().AddAppliedSchema("PhysxConvexHullCollisionAPI")
            mesh.GetPrim().CreateAttribute("physxConvexHullCollision:minThickness",
                                           Sdf.ValueTypeNames.Float).Set(CONVEX_MIN_THICKNESS)
            gprim = mesh
        prim = gprim.GetPrim()
        gprim.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(REST_OFFSET)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    stage.GetRootLayer().Save()
    return entry


# ---------------------------------------------------------------------------------------------
# Isaac Lab helpers
# ---------------------------------------------------------------------------------------------

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "usd" / "fasteners"


def fastener_usd_path(name: str, out_dir: str | Path | None = None) -> str:
    """Absolute path of a generated fastener, e.g. fastener_usd_path('hex_bolt_M6x20')."""
    path = Path(out_dir or DEFAULT_OUT_DIR) / f"{name}.usda"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python isaac_envs/assets/fasteners.py --only {name}`")
    return str(path)


def fastener_spawn_cfg(name: str, **kwargs):
    """``sim_utils.UsdFileCfg`` for a generated fastener (pass rigid_props=..., etc. to override).

    A list of these is a valid ``MultiAssetSpawnerCfg(assets_cfg=[...])``.
    """
    import isaaclab.sim as sim_utils  # noqa: PLC0415

    return sim_utils.UsdFileCfg(usd_path=fastener_usd_path(name), **kwargs)


def generate(specs: list[FastenerSpec] | None = None, out_dir: str | Path | None = None,
             round_segments: int | None = None) -> list[dict]:
    out_dir = Path(out_dir or DEFAULT_OUT_DIR)
    specs = specs or default_specs()
    entries = [write_usd(spec, out_dir / f"{spec.name}.usda", round_segments) for spec in specs]
    manifest_path = out_dir / "manifest.json"
    old = {}
    if manifest_path.exists():
        old = {e["name"]: e for e in json.loads(manifest_path.read_text())["fasteners"]}
    old.update({e["name"]: e for e in entries})
    order = [s.name for s in default_specs()]
    merged = sorted(old.values(), key=lambda e: order.index(e["name"]) if e["name"] in order else len(order))
    manifest = {
        "generator": "isaac_envs/assets/fasteners.py",
        "units": {"length": "mm", "mass": "g", "angle": "rad", "com": "m (USD frame)"},
        "density_kg_m3": STEEL_DENSITY,
        "jaw_contact_angle": "2*asin(grasp_width_m/0.186)",
        "grasp_width": "shank major diameter (screws), across flats (nuts), outside diameter (washers)",
        "reference_mass": REFERENCE_SOURCE,
        "fasteners": merged,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return entries


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="e.g. hex_bolt_M6x20 washer_M6")
    parser.add_argument("--list", action="store_true", help="print the set with masses; write nothing")
    parser.add_argument("--round-segments", type=int, default=None,
                        help=f"round colliders: 0 = cylinder primitives, N = convex N-gon (default {ROUND_COLLIDER_SEGMENTS})")
    args = parser.parse_args(argv)
    specs = [spec_from_name(n) for n in args.only] if args.only else default_specs()
    if args.list:
        for spec in specs:
            e = manifest_entry(spec)
            ref = f"{e['ref_mass_g']:7.3f} ({e['ref_mass_err_pct']:+.1f}%)" if e["ref_mass_g"] else "      -"
            print(f"{e['name']:22s} {e['mass_g']:8.3f} g  ref {ref}  grip {e['grasp_width_mm']:5.1f} mm"
                  f"  angle {e['jaw_contact_angle_rad']:.4f}")
        return
    for e in generate(specs, args.out, args.round_segments):
        print(f"wrote {e['usd']}  {e['mass_g']:.3f} g")


if __name__ == "__main__":
    _main()
