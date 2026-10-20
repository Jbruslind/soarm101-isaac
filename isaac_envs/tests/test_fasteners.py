"""Unit tests for isaac_envs/assets/fasteners.py (no Isaac needed; the USD test needs only pxr).

Run:  python -m pytest isaac_envs/tests   (host or inside the Isaac image with /isaac-sim/python.sh -m pytest)
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # isaac_envs/, like the env scripts

from assets import fasteners as F  # noqa: E402

M6X20 = F.FastenerSpec("hex_bolt", "M6", 20)


# --- dimension tables ---------------------------------------------------------------------------


def test_iso4017_nominal_dims():
    # ISO 4017 grade A: s nom = max, k nom
    assert [F.ISO4017[s]["s"] for s in F.SIZES] == [5.5, 7.0, 8.0, 10.0, 13.0]
    assert [F.ISO4017[s]["k"] for s in F.SIZES] == [2.0, 2.8, 3.5, 4.0, 5.3]


def test_iso4762_nominal_dims():
    assert [F.ISO4762[s]["dk"] for s in F.SIZES] == [5.5, 7.0, 8.5, 10.0, 13.0]
    assert [F.ISO4762[s]["k"] for s in F.SIZES] == [F.NOMINAL_D[s] for s in F.SIZES]  # k = d
    assert [F.ISO4762[s]["s"] for s in F.SIZES] == [2.5, 3.0, 4.0, 5.0, 6.0]
    assert [F.ISO4762[s]["b"] for s in F.SIZES] == [2 * F.NOMINAL_D[s] + 12 for s in F.SIZES]


def test_iso4032_and_iso7089_dims():
    assert [F.ISO4032[s]["m"] for s in F.SIZES] == [2.4, 3.2, 4.7, 5.2, 6.8]
    assert [F.ISO4032[s]["s"] for s in F.SIZES] == [F.ISO4017[s]["s"] for s in F.SIZES]
    assert [(F.ISO7089[s]["d1"], F.ISO7089[s]["d2"], F.ISO7089[s]["h"]) for s in F.SIZES] == [
        (3.2, 7.0, 0.5), (4.3, 9.0, 0.8), (5.3, 10.0, 1.0), (6.4, 12.0, 1.6), (8.4, 16.0, 1.6)]


def test_thread_diameters_iso724():
    assert F.pitch_diameter("M6") == pytest.approx(5.350, abs=1e-3)
    assert F.pitch_diameter("M8") == pytest.approx(7.188, abs=1e-3)
    assert F.minor_diameter_internal("M6") == pytest.approx(4.917, abs=1e-3)
    assert F.minor_diameter_external("M6") == pytest.approx(4.773, abs=1e-3)


def test_lengths_inside_standard_ranges():
    ranges4017 = {"M3": (6, 30), "M4": (8, 40), "M5": (10, 50), "M6": (12, 60), "M8": (16, 80)}
    ranges4762 = {"M3": (5, 30), "M4": (6, 40), "M5": (8, 50), "M6": (10, 60), "M8": (12, 80)}
    for size in F.SIZES:
        lo, hi = ranges4017[size]
        assert all(lo <= l <= hi for l in F.ISO4017_LENGTHS[size])
        lo, hi = ranges4762[size]
        assert all(lo <= l <= hi for l in F.ISO4762_LENGTHS[size])


def test_default_set():
    specs = F.default_specs()
    names = [s.name for s in specs]
    assert names[0] == "hex_bolt_M6x20"
    assert len(names) == len(set(names))
    assert Counter(s.kind for s in specs) == {"hex_bolt": 28, "socket_cap": 31, "hex_nut": 5, "washer": 5}
    for s in specs:
        assert F.spec_from_name(s.name) == s


# --- mesh and mass maths ------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["hex_bolt_M6x20", "socket_cap_M8x40", "hex_nut_M6", "washer_M3"])
@pytest.mark.parametrize("mass_model", [False, True])
def test_mesh_is_closed_and_outward(name, mass_model):
    mesh = F.loft_mesh(F.spec_from_name(name), 48, mass_model)
    edges = Counter()
    for f in mesh.faces:
        assert len(set(f)) == len(f) >= 3
        for a, b in zip(f, f[1:] + f[:1]):
            edges[(a, b)] += 1
    # every directed edge exactly once and its reverse exactly once -> closed, consistently oriented
    assert all(n == 1 for n in edges.values())
    assert all(edges[(b, a)] == 1 for a, b in edges)
    assert F.mass_properties(mesh, 1.0)["volume"] > 0


def test_washer_mass_and_inertia_match_annulus():
    spec = F.FastenerSpec("washer", "M6")
    d = spec.dims
    r1, r2, h = d["d1"] / 2e3, d["d2"] / 2e3, d["h"] / 1e3
    mass = F.STEEL_DENSITY * math.pi * (r2**2 - r1**2) * h
    props = F.fastener_mass_properties(spec)
    assert props["mass"] == pytest.approx(mass, rel=2e-4)  # 360-gon vs circle
    izz = 0.5 * mass * (r1**2 + r2**2)
    ixx = mass / 12.0 * (3 * (r1**2 + r2**2) + h**2)
    assert props["inertia"][2][2] == pytest.approx(izz, rel=1e-3)
    assert props["inertia"][0][0] == pytest.approx(ixx, rel=1e-3)
    assert props["com"][2] == pytest.approx(h / 2, rel=1e-6)


def test_hex_prism_volume_exact():
    # a hex nut's mass mesh minus the d2 bore, without chamfers, is a hexagonal prism: check the
    # integrator on the unchamfered hexagon area s^2 sqrt(3)/2 via the collision hull helper
    pts = F._prism(F._hexagon(10.0), 0.0, 4.0)
    n = 6
    faces = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))] + [[i, (i + 1) % n, n + (i + 1) % n, n + i]
                                                                     for i in range(n)]
    vol = F.mass_properties(F.TriMesh(pts, faces), 1.0)["volume"]
    assert vol == pytest.approx(math.sqrt(3) / 2 * 100 * 4, rel=1e-12)


def test_m6x20_hex_mass():
    # DIN 933 (= ISO 4017 dims for M6) weight table: 5.80 kg / 1000 pcs. Our nominal-geometry model
    # (s = 10, k = 4, thread at d2) gives 6.13 g, +5.6 %: the DIN head (implied 2.37 g, 302 mm^3) is
    # 13 % below a nominal 10 x 4 mm hex prism (346 mm^3), more than any chamfer explains.
    mass_g = F.fastener_mass_properties(M6X20)["mass"] * 1e3
    assert 5.6 <= mass_g <= 6.2
    assert mass_g == pytest.approx(5.80, rel=0.06)


def test_m6x20_hand_calculation():
    # hex prism (s=10, k=4) + d2 rod (5.350 x 20) = 346.4 + 449.6 mm^3 = 6.25 g before chamfers,
    # washer face and the ISO 4753 point; those remove 2 %.
    hand = 7.85e-3 * (math.sqrt(3) / 2 * 10**2 * 4 + math.pi / 4 * F.pitch_diameter("M6") ** 2 * 20)
    mass_g = F.fastener_mass_properties(M6X20)["mass"] * 1e3
    assert hand == pytest.approx(6.245, abs=0.01)
    assert 0.97 * hand < mass_g < hand


def test_thread_increment_matches_din_tables():
    # mass per mm of thread = rho * pi/4 * d2^2; DIN 933 table increments (g/mm): M5 0.124, M6 0.172, M8 0.320
    for size, table in (("M5", 0.124), ("M6", 0.172), ("M8", 0.320)):
        a = F.fastener_mass_properties(F.FastenerSpec("hex_bolt", size, 20))["mass"]
        b = F.fastener_mass_properties(F.FastenerSpec("hex_bolt", size, 30))["mass"]
        assert (b - a) * 1e3 / 10 == pytest.approx(table, rel=0.03)


def test_masses_against_reference_tables():
    errs = {}
    for spec in F.default_specs():
        ref = F.reference_mass_g(spec)
        if ref:
            errs[spec.name] = F.fastener_mass_properties(spec)["mass"] * 1e3 / ref - 1.0
    assert len(errs) >= 60
    within5 = [n for n, e in errs.items() if abs(e) <= 0.05]
    # The DIN weight tables are "approximate, commercial items" and not self-consistent (e.g. DIN 912
    # lists M6x100 lighter than M6x90, and its M5x10 implies a head lighter than 2/3 of a 8.5 x 5 mm
    # cylinder). Require: no part off by more than 16 %, 70 % within 5 % (47 of 64 today), all nuts,
    # washers, M4/M5/M8 hex bolts and M6/M8 SHCS within 5 %.
    assert max(abs(e) for e in errs.values()) < 0.16
    assert len(within5) >= 0.70 * len(errs)
    for name, e in errs.items():
        spec = F.spec_from_name(name)
        if spec.kind in ("hex_nut", "washer") or (spec.kind == "hex_bolt" and spec.size in ("M4", "M5", "M8")) \
                or (spec.kind == "socket_cap" and spec.size in ("M6", "M8")):
            assert abs(e) <= 0.05, (name, e)


def test_com_and_inertia_sane():
    props = F.fastener_mass_properties(M6X20)
    cx, cy, cz = props["com"]
    assert abs(cx) < 1e-9 and abs(cy) < 1e-9
    assert -0.010 < cz < 0.0  # between the bearing face and the shank middle
    I = props["inertia"]
    assert I[0][0] == pytest.approx(I[1][1], rel=1e-6)
    assert max(abs(I[0][1]), abs(I[0][2]), abs(I[1][2])) < 1e-6 * I[0][0]
    assert I[2][2] < I[0][0]  # slender: easiest to spin about its axis


def test_collision_shapes_fit_nominal_envelope():
    expected_r = {"hex_bolt_M6x20": 10 / math.sqrt(3), "socket_cap_M8x40": 6.5, "hex_nut_M6": 10 / math.sqrt(3),
                  "washer_M6": 6.0, "washer_M3": 3.5}
    for name, r_exp in expected_r.items():
        spec = F.spec_from_name(name)
        pts = [(x * 1e3, y * 1e3, z * 1e3) for x, y, z in F.collision_support_points(spec)]
        assert max(math.hypot(x, y) for x, y, _ in pts) == pytest.approx(r_exp, rel=1e-9)
        zs = [z for _, _, z in pts]
        assert min(zs) == pytest.approx(-spec.length if spec.is_screw else 0.0)
        assert max(zs) == pytest.approx(spec.head_height)


def test_thin_washer_uses_box_colliders():
    kinds = {name: {s[1] for s in F.collision_shapes(F.spec_from_name(name))} for name in ("washer_M3", "washer_M4")}
    assert kinds == {"washer_M3": {"box"}, "washer_M4": {"convex"}}


# --- grasp ---------------------------------------------------------------------------------------


def test_grasp_width_and_jaw_angle():
    assert M6X20.grasp_width == 6.0
    assert F.FastenerSpec("hex_nut", "M6").grasp_width == 10.0
    assert F.FastenerSpec("washer", "M8").grasp_width == 16.0
    assert F.jaw_contact_angle(6.0) == pytest.approx(2 * math.asin(0.006 / 0.186), rel=1e-12)
    assert F.jaw_contact_angle(6.0) == pytest.approx(0.06453, abs=1e-5)


# --- committed assets ----------------------------------------------------------------------------


def test_committed_manifest_matches_generator():
    manifest = json.loads((F.DEFAULT_OUT_DIR / "manifest.json").read_text())
    entries = {e["name"]: e for e in manifest["fasteners"]}
    assert set(entries) == {s.name for s in F.default_specs()}
    for spec in F.default_specs():
        e = entries[spec.name]
        assert e == F.manifest_entry(spec), spec.name
        assert (F.DEFAULT_OUT_DIR / e["usd"]).is_file()
        for key in ("type", "size", "length_mm", "mass_g", "grasp_width_mm", "jaw_contact_angle_rad"):
            assert key in e


def test_usd_rigid_body(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    path = tmp_path / "m6x20.usda"
    F.write_usd(M6X20, path)
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    assert root.GetPath().pathString == "/Fastener"
    assert UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert root.HasAPI(UsdPhysics.RigidBodyAPI) and root.HasAPI(UsdPhysics.MassAPI)
    assert UsdPhysics.MassAPI(root).GetMassAttr().Get() == pytest.approx(
        F.fastener_mass_properties(M6X20)["mass"], rel=1e-6)
    colliders = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert {p.GetName() for p in colliders} == {"shank", "head"}
    for p in colliders:
        assert p.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.001)
        assert p.GetAttribute("physxCollision:restOffset").Get() == 0.0
    for p in colliders:
        assert UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == "convexHull"
        assert p.GetAttribute("physxConvexHullCollision:minThickness").Get() <= 1e-4
    # exactly one rigid body (Isaac Lab RigidObject requirement)
    assert sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(root)) == 1
    vis = UsdGeom.Mesh(stage.GetPrimAtPath("/Fastener/visual"))
    lo, hi = vis.GetExtentAttr().Get()
    assert hi[2] == pytest.approx(0.004) and lo[2] == pytest.approx(-0.020)
    assert hi[0] == pytest.approx(0.005)  # across flats / 2
