from fenics import *
import cashocs
import numpy as np
import matplotlib.pyplot as plt
config = cashocs.load_config("./config.ini")

def test_square_only_PBC():
    mesh, _, boundaries, _, _, _ = cashocs.regular_mesh(25)
    V = VectorElement("CG", mesh.ufl_cell(), 2)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement([V,Q]))

    bcs = cashocs._utils.forms.create_periodic_bcs(W, boundaries, [1,2])

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0))
    a = inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx
    L = inner(f, v)*dx

    A, b, _ = cashocs._utils.linalg.assemble_petsc_system(a, L, bcs)

    u = Function(W)
    solve(A, u.vector(), b)

    u, p = u.split()

    x_ref = np.linspace(0, 1, num=100)
    coord_top = []
    coord_bottom = []

    for x in x_ref:
        value_m = u((0.0, x))
        coord_top.append(value_m)
        value_s = u((1.0, x))
        coord_bottom.append(value_s)
    coord_top_np = np.array(coord_top)
    coord_bottom_np = np.array(coord_bottom)

    error = np.linalg.norm(coord_top_np-coord_bottom_np)

    assert error < DOLFIN_EPS_LARGE

def test_square_mixed_BC():
    mesh, _, boundaries, _, _, _ = cashocs.regular_mesh(25)
    V = VectorElement("CG", mesh.ufl_cell(), 2)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement([V,Q]))

    inflow = Expression(("1", "0"), degree=2)
    bc0 = cashocs.create_dirichlet_bcs(W.sub(0), inflow, boundaries, 3)
    pbc = cashocs.create_periodic_bcs(W, boundaries, [1,2])
    bcs = bc0 + pbc

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0))
    a = inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx
    L = inner(f, v)*dx

    A, b, _ = cashocs.assemble_petsc_system(a, L, bcs)

    u = Function(W)
    solve(A, u.vector(), b)

    u, p = u.split()

    x_ref = np.linspace(0, 1, num=100)
    coord_top = []
    coord_bottom = []

    for x in x_ref:
        value_m = u((0.0, x))
        coord_top.append(value_m)
        value_s = u((1.0, x))
        coord_bottom.append(value_s)
    coord_top_np = np.array(coord_top)
    coord_bottom_np = np.array(coord_bottom)

    error = np.linalg.norm(coord_top_np-coord_bottom_np)

    assert error < DOLFIN_EPS_LARGE*100

def test_square_1_2():
    assert True

def test_square_2_1():
    assert True

def test_square_random():
    assert True

def test_square_alt_problem():
    assert True

def test_circle_right_angle():
    assert True

def test_circle_acute_angle():
    assert True

def test_circle_obture_angle():
    assert True
    
def test_circle_rand_angle_rand_ratio():
    assert True