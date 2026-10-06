"""Tests that function values on the main and secondary periodic boundaries
match within a tolerance, for different problem types (Poisson, Stokes),
function spaces (scalar, vector, mixed), geometries/angles, and for
optimal-control and shape-optimization problems.

The meshes come from ``cashocs.regular_mesh`` (tags: 1=left, 2=right,
3=bottom, 4=top). Periodic BCs are imposed via ``cashocs.create_periodic_bcs``.
"""

import os

import numpy as np
import pytest
import fenics
from petsc4py import PETSc
from cashocs import _utils
import cashocs
import matplotlib.pyplot as plt

TAG_LEFT, TAG_RIGHT, TAG_BOTTOM, TAG_TOP = 1, 2, 3, 4

CONFIG_OCP = os.path.join(os.path.dirname(__file__), "config_ocp.ini")
CONFIG_SOP = os.path.join(os.path.dirname(__file__), "config_sop.ini")


class ShiftPeriodic(fenics.SubDomain):
    """Map the main boundary to the secondary one by a translation."""

    def __init__(self, shift, main_coord, axis, **kw):
        super().__init__(**kw)
        self.shift = np.asarray(shift, float)
        self.main_coord = main_coord
        self.axis = axis

    def inside(self, x, on_boundary):
        return bool(on_boundary and fenics.near(x[self.axis], self.main_coord))

    def map(self, x, y):
        y[:] = np.asarray(x) + self.shift


class ShearedPeriodic(fenics.SubDomain):
    """Map sheared (oblique) left boundary onto the right one; the shear
    parameter mimics an angle between the periodic direction and the x-axis."""

    def __init__(self, shear, length_x, **kw):
        super().__init__(**kw)
        self.shear = shear
        self.length_x = length_x

    def inside(self, x, on_boundary):
        return bool(
            on_boundary and fenics.near(x[0] - self.shear * x[1], 0.0, 1e-6)
        )

    def map(self, x, y):
        y[0] = x[0] - self.length_x
        y[1] = x[1]


def shear_mesh(mesh, shear):
    mesh.coordinates()[:, 0] += shear * mesh.coordinates()[:, 1]
    mesh.bounding_box_tree().build(mesh)
    return mesh


def solve_with_ksp(A: PETSc.Mat, b: PETSc.Vec) -> PETSc.Vec:
    """Solve A x = b with a direct PETSc KSP solver (LU/MUMPS)."""
    x = b.duplicate()
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    ksp.solve(b, x)
    x.assemble()
    return x


def assemble_petsc(a_form, L_form, bcs=None):
    """Assemble bilinear/linear forms into PETSc.Mat / PETSc.Vec."""
    A = fenics.PETScMatrix()
    b = fenics.PETScVector()
    fenics.assemble(a_form, tensor=A)
    fenics.assemble(L_form, tensor=b)
    if bcs is not None:
        for bc in bcs:
            bc.apply(A, b)
    return A, b


def compare_boundary_values(u, main_points, secondary_map, atol=1e-6):
    """Assert that u(main) == u(secondary) for each sampled point."""
    for p in main_points:
        val_main = np.atleast_1d(u(fenics.Point(*p)))
        val_sec = np.atleast_1d(u(fenics.Point(*secondary_map(p))))
        assert np.allclose(val_main, val_sec, atol=atol), (
            f"Mismatch at main={p} ({val_main}) vs secondary="
            f"{secondary_map(p)} ({val_sec})"
        )


def sample_line(axis, fixed, n=7, rng=(0.05, 0.95)):
    return [tuple(fixed if i == axis else t for i in range(2))
            for t in np.linspace(*rng, n)]


@pytest.mark.parametrize("order", [1, 2])
def test_poisson_scalar_periodic(order):
    mesh, _, boundaries, dx, ds, dS = cashocs.regular_mesh(20)
    cd = ShiftPeriodic(shift=(1.0, 0.0), main_coord=0.0, axis=0)
    V = fenics.FunctionSpace(mesh, "CG", order, constrained_domain=cd)

    u, v = fenics.TrialFunction(V), fenics.TestFunction(V)
    f = fenics.Expression("sin(2*pi*x[1]) + x[0]*x[0]", degree=2)
    a = fenics.inner(fenics.grad(u), fenics.grad(v)) * dx + u * v * dx
    L = f * v * dx

    bcs = [fenics.DirichletBC(V, fenics.Constant(0.0), boundaries, TAG_BOTTOM)]
    bcs += [fenics.DirichletBC(V, fenics.Constant(0.0), boundaries, TAG_TOP)]
    bcs += _utils.create_periodic_bcs(V, boundaries, [TAG_LEFT, TAG_RIGHT], cd)

    u_sol = fenics.Function(V)
    _utils.assemble_and_solve_linear(a, L, u_sol, bcs)

    compare_boundary_values(u_sol, sample_line(0, 0.0),
                            lambda p: (1.0, p[1]), atol=1e-2)


def test_elasticity_vector_periodic():
    mesh, _, boundaries, dx, ds, dS = cashocs.regular_mesh(20)
    cd = ShiftPeriodic(shift=(0.0, 1.0), main_coord=0.0, axis=1)
    V = fenics.VectorFunctionSpace(mesh, "CG", 1, dim=2,
                                   constrained_domain=cd)

    u, v = fenics.TrialFunction(V), fenics.TestFunction(V)
    eps = lambda w: fenics.sym(fenics.grad(w))
    sigma = lambda w: fenics.tr(eps(w)) * fenics.Identity(2) + 2 * eps(w)
    a = fenics.inner(sigma(u), eps(v)) * dx
    L = fenics.inner(fenics.Constant((0.0, -1.0)), v) * dx

    bcs = [fenics.DirichletBC(V, fenics.Constant((0.0, 0.0)), boundaries, TAG_LEFT)]
    bcs += _utils.create_periodic_bcs(V, boundaries, [TAG_BOTTOM, TAG_TOP], cd)

    u_sol = fenics.Function(V)
    _utils.assemble_and_solve_linear(a, L, u_sol, bcs)

    compare_boundary_values(u_sol, sample_line(1, 0.0),
                            lambda p: (p[0], 1.0), atol=1e-2)


def test_stokes_mixed_periodic():
    mesh, _, boundaries, dx, ds, dS = cashocs.regular_mesh(20)
    cd = ShiftPeriodic(shift=(1.0, 0.0), main_coord=0.0, axis=0)
    TH = fenics.MixedElement(
        [fenics.VectorElement("CG", mesh.ufl_cell(), 2),
         fenics.FiniteElement("CG", mesh.ufl_cell(), 1)])
    W = fenics.FunctionSpace(mesh, TH)#, constrained_domain=cd)

    (u, p), (v, q) = fenics.TrialFunctions(W), fenics.TestFunctions(W)
    a = (fenics.inner(fenics.grad(u), fenics.grad(v)) * dx
         - p * fenics.div(v) * dx - q * fenics.div(u) * dx)
    L = fenics.inner(fenics.Constant((0.0, -1.0)), v) * dx

    bcs = [fenics.DirichletBC(W.sub(0), fenics.Constant((0.0, 0.0)),
                              boundaries, TAG_BOTTOM),
           fenics.DirichletBC(W.sub(0), fenics.Constant((1.0, 0.0)),
                              boundaries, TAG_TOP)]
    bcs += _utils.create_periodic_bcs(W, boundaries, [TAG_LEFT, TAG_RIGHT], cd)

    w_sol = fenics.Function(W)
    _utils.assemble_and_solve_linear(a, L, w_sol, bcs)
    u_sol, p_sol = w_sol.split(deepcopy=True)

    mp = sample_line(0, 0.0)
    sec = lambda p: (1.0, p[1])
    compare_boundary_values(u_sol, mp, sec, atol=1e-2)
    compare_boundary_values(p_sol, mp, sec, atol=1e-2)


@pytest.mark.parametrize("shear", [0.0, 0.3])
def test_poisson_scalar_rotated_periodic(shear):
    mesh, _, boundaries, dx, ds, dS = cashocs.regular_mesh(20)
    mesh = shear_mesh(mesh, shear)
    cd = ShearedPeriodic(shear=shear, length_x=1.0)
    V = fenics.FunctionSpace(mesh, "CG", 1, constrained_domain=cd)

    u, v = fenics.TrialFunction(V), fenics.TestFunction(V)
    a = fenics.inner(fenics.grad(u), fenics.grad(v)) * dx + u * v * dx
    L = fenics.Expression("1.0 + 0.5*x[1]", degree=1) * v * dx

    bcs = [fenics.DirichletBC(V, fenics.Constant(0.0), boundaries, TAG_BOTTOM)]

    # A sheared interface is "curved" in the mapped frame, so cashocs's
    # assemble_and_solve_linear rejects it. Apply the periodic BCs directly
    # on the assembled PETSc system via PeriodicBoundaryInterpolator instead.
    A, b = assemble_petsc(a, L, bcs=bcs)
    pbc = _utils.PeriodicBC(
        V, boundaries, main=TAG_LEFT, secondary=TAG_RIGHT,
        constrained_domain=cd,
    )
    pbi = _utils.PeriodicBoundaryInterpolator(pbc)
    A_mod, b_mod = pbi.apply_periodic_bcs(A), pbi.apply_periodic_bcs(b)
    x = solve_with_ksp(A_mod, b_mod)

    u_sol = fenics.Function(V)
    u_sol.vector().set_local(x.getArray())
    u_sol.vector().apply("insert")

    ys = np.linspace(0.05, 0.95, 7)
    mp = [(shear * y, y) for y in ys]
    compare_boundary_values(u_sol, mp, lambda p: (1.0 + shear * p[1], p[1]),
                            atol=1e-6)


def test_control_problem_periodic():
    mesh, _, boundaries, dx, ds, dS = cashocs.regular_mesh(20)
    cd = ShiftPeriodic(shift=(1.0, 0.0), main_coord=0.0, axis=0)
    V = fenics.FunctionSpace(mesh, "CG", 1, constrained_domain=cd)

    y, p, u = (fenics.Function(V) for _ in range(3))
    F = (fenics.inner(fenics.grad(y), fenics.grad(p)) * dx
         + y * p * dx - u * p * dx)
    bcs = [fenics.DirichletBC(V, fenics.Constant(0.0), boundaries, TAG_BOTTOM)]
    #bcs += cashocs.create_periodic_bcs(V, boundaries, [TAG_LEFT, TAG_RIGHT], cd)

    y_d = fenics.Expression("sin(2*pi*x[1])", degree=4)
    J = cashocs.IntegralFunctional(
        fenics.Constant(0.5) * (y - y_d) * (y - y_d) * dx
        + fenics.Constant(1e-4) * u * u * dx)

    ocp = cashocs.OptimalControlProblem(
        F, bcs, J, y, u, p,
        config=cashocs.load_config(CONFIG_OCP))
    ocp.solve(algorithm="lbfgs", rtol=1e-1, max_iter=100)

    compare_boundary_values(y, sample_line(0, 0.0), lambda pt: (1.0, pt[1]),
                            atol=1e-2)


def _boundary_vertex_ids(mesh, boundaries, tags):
    """Return vertex IDs belonging to facets with one of the given tags."""
    tags = set(tags)
    tdim = mesh.topology().dim()
    mesh.init(tdim - 1, 0)

    vertex_ids = set()
    for facet in fenics.facets(mesh):
        if int(boundaries[facet]) in tags:
            vertex_ids.update(facet.entities(0))

    return np.asarray(sorted(vertex_ids), dtype=np.intp)


def test_shape_problem_periodic_boundaries_fixed():
    mesh, _, boundaries, dx, ds, dS = cashocs.regular_mesh(20)

    cd = ShiftPeriodic(shift=(0.0, 1.0), main_coord=0.0, axis=1)
    periodic_vertex_ids = _boundary_vertex_ids(mesh, boundaries, [TAG_BOTTOM, TAG_TOP])
    periodic_coordinates_before = (mesh.coordinates()[periodic_vertex_ids].copy())

    V = fenics.FunctionSpace(mesh, "CG", 1, constrained_domain=cd)
    y = fenics.Function(V)
    p = fenics.Function(V)

    F = (fenics.inner(fenics.grad(y), fenics.grad(p)) * dx
        - fenics.Constant(1.0) * p * dx)

    bcs = [fenics.DirichletBC(V, fenics.Constant(0.0), boundaries, TAG_LEFT)]
    bcs += cashocs.create_periodic_bcs(V, boundaries, [TAG_BOTTOM, TAG_TOP], cd)

    J = cashocs.IntegralFunctional(y * dx)

    config = cashocs.load_config(CONFIG_SOP)

    config.set("Mesh", "remesh", "False")

    config.set("ShapeGradient", "shape_bdry_fix", str([TAG_LEFT, TAG_BOTTOM, TAG_TOP]))
    config.set("ShapeGradient", "shape_bdry_def", str([TAG_RIGHT]))

    sop = cashocs.ShapeOptimizationProblem(F, bcs, J, y, p, boundaries, config=config)

    sop.solve(algorithm="lbfgs", rtol=1e-1, max_iter=100)

    compare_boundary_values(y, sample_line(1, 0.0), lambda pt: (pt[0], 1.0),
                            atol=1e-3)
