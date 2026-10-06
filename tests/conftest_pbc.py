"""Shared fixtures for the PeriodicBoundaryInterpolator test suite."""

from __future__ import annotations

import dataclasses
import typing

import fenics
import numpy as np
import pytest

# ------------------------------------------------------------------ hooks
# Everything version-specific lives here.
MAP_MAIN_TO_SECONDARY = True   # flip if test_map_orientation fails
TAG_MAIN, TAG_SECONDARY = 1, 2
TAG_BOTTOM, TAG_TOP = 3, 4

try:
    from cashocs._forms import PeriodicBC as _PeriodicBC
except Exception:                                    # pragma: no cover
    _PeriodicBC = None


class _PBCStub:
    """Stand-in exposing exactly the attributes the interpolator reads."""

    def __init__(self, boundaries, functionspace, constrained_domain,
                 main_idc, secondary_idc):
        self.boundaries = boundaries
        self.functionspace = functionspace
        self.constrained_domain = constrained_domain
        self.main_idc = main_idc
        self.secondary_idc = secondary_idc


def make_pbc(boundaries, V, cd, main=TAG_MAIN, secondary=TAG_SECONDARY):
    if _PeriodicBC is None:
        return _PBCStub(boundaries, V, cd, main, secondary)
    try:
        return _PeriodicBC(boundaries, V, cd, main, secondary)
    except TypeError:
        return _PBCStub(boundaries, V, cd, main, secondary)


# ------------------------------------------------------------- geometry
@dataclasses.dataclass
class Geometry:
    mesh: fenics.Mesh
    boundaries: fenics.MeshFunction
    shift: np.ndarray          # main + shift == secondary
    label: str


def _affine(angle_deg: float, kind: str) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    if kind == "none":
        return np.eye(2)
    if kind == "shear":                     # parallelogram, periodic dir stays e_x
        return np.array([[1.0, np.tan(a)], [0.0, 1.0]])
    if kind == "rotate":                    # periodic direction becomes oblique
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    if kind == "stretch":
        return np.diag([1.0, 1.0 + angle_deg / 45.0])
    raise ValueError(kind)


def build_geometry(n: int = 12, angle_deg: float = 0.0, kind: str = "none",
                   refine_left: bool = False) -> Geometry:
    """Unit square, marked *before* the affine map so tags stay exact."""
    mesh = fenics.UnitSquareMesh(n, n)

    if refine_left:                          # -> non-matching interface meshes
        marker = fenics.MeshFunction("bool", mesh, mesh.topology().dim(), False)
        for cell in fenics.cells(mesh):
            marker[cell] = cell.midpoint().x() < 0.25
        mesh = fenics.refine(mesh, marker)

    boundaries = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    fenics.CompiledSubDomain("near(x[0], 0.0) && on_boundary").mark(boundaries, TAG_MAIN)
    fenics.CompiledSubDomain("near(x[0], 1.0) && on_boundary").mark(boundaries, TAG_SECONDARY)
    fenics.CompiledSubDomain("near(x[1], 0.0) && on_boundary").mark(boundaries, TAG_BOTTOM)
    fenics.CompiledSubDomain("near(x[1], 1.0) && on_boundary").mark(boundaries, TAG_TOP)

    T = _affine(angle_deg, kind)
    xy = mesh.coordinates()
    xy[:] = xy @ T.T
    mesh.bounding_box_tree().build(mesh)

    return Geometry(mesh, boundaries, T @ np.array([1.0, 0.0]),
                    f"{kind}{angle_deg:g}{'-ref' if refine_left else ''}")


# ------------------------------------------------------- periodic maps
class CashocsMap(fenics.SubDomain):
    """`map` in the direction the interpolator expects."""

    def __init__(self, shift, tol=1e-10):
        super().__init__()
        self.s = np.asarray(shift, float)
        self.tol = tol

    def inside(self, x, on_boundary):
        return bool(on_boundary and abs(x[0] - self.s[1] * x[1] / max(self.s[1], 1e-30)
                                        if False else
                                        self._param(x)) < self.tol)

    def _param(self, x):
        # signed distance along `shift` from the main edge through the origin
        n = np.array([self.s[1], -self.s[0]])
        n /= np.linalg.norm(n)
        return np.dot(np.asarray(x[:2]), n)

    def map(self, x, y):
        sign = 1.0 if MAP_MAIN_TO_SECONDARY else -1.0
        y[0] = x[0] + sign * self.s[0]
        y[1] = x[1] + sign * self.s[1]


class FenicsPBC(fenics.SubDomain):
    """FEniCS convention: `inside` marks the master, `map` sends slave -> master."""

    def __init__(self, shift, tol=1e-10):
        super().__init__()
        self.s = np.asarray(shift, float)
        self.tol = tol
        self._n = np.array([shift[1], -shift[0]])
        self._n /= np.linalg.norm(self._n)

    def inside(self, x, on_boundary):
        return bool(on_boundary and abs(np.dot(np.asarray(x[:2]), self._n)) < self.tol)

    def map(self, x, y):
        y[0] = x[0] - self.s[0]
        y[1] = x[1] - self.s[1]


# ------------------------------------------------------------- fixtures
@pytest.fixture(params=[
    dict(kind="none", angle_deg=0.0),
    dict(kind="shear", angle_deg=30.0),
    dict(kind="rotate", angle_deg=25.0),
    dict(kind="stretch", angle_deg=15.0),
], ids=lambda p: f"{p['kind']}{p['angle_deg']:g}")
def geometry(request):
    return build_geometry(n=12, **request.param)


@pytest.fixture
def square():
    return build_geometry(n=10)


@pytest.fixture
def nonmatching():
    return build_geometry(n=8, refine_left=True)


@pytest.fixture
def interpolator_cls():
    from cashocs._utils import PeriodicBoundaryInterpolator   # adapt
    return PeriodicBoundaryInterpolator


# --------------------------------------------------------------- helpers
def interior_points(geo: Geometry, k: int = 7) -> np.ndarray:
    """Points safely inside the (possibly skewed) domain."""
    u, v = np.meshgrid(np.linspace(0.15, 0.85, k), np.linspace(0.15, 0.85, k))
    ref = np.column_stack([u.ravel(), v.ravel()])
    xy = geo.mesh.coordinates()
    # recover the affine map from the four corners of the reference square
    A = np.linalg.lstsq(np.array([[0, 0], [1, 0], [0, 1]]),
                        xy[:3] * 0 + xy[:3], rcond=None)[0] if False else None
    # simpler: bounding-box-free evaluation via the stored shift + mesh extent
    return ref @ np.column_stack([geo.shift, _orth(geo)]).T


def _orth(geo: Geometry) -> np.ndarray:
    xy = geo.mesh.coordinates()
    n = np.array([-geo.shift[1], geo.shift[0]])
    n /= np.linalg.norm(n)
    h = (xy @ n).max() - (xy @ n).min()
    return n * h


def sample(fn: fenics.Function, pts: np.ndarray) -> np.ndarray:
    out = []
    for p in pts:
        try:
            out.append(np.atleast_1d(fn(p)).copy())
        except RuntimeError:                       # point outside -> skip
            continue
    assert out, "no evaluation point inside the mesh"
    return np.array(out)


def petsc_dense(mat) -> np.ndarray:
    m = mat.mat() if hasattr(mat, "mat") else mat
    return m[:, :]