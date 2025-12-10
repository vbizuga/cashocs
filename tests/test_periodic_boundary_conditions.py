from fenics import *
import cashocs
import numpy as np

def test_square_only_PBC():
    mesh, _, boundaries, _, _, _ = cashocs.regular_mesh(25)
    V = VectorElement("CG", mesh.ufl_cell(), 2)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement([V,Q]))

    class PeriodicBoundary(SubDomain):
        def map(self, x, y):
            x[0] = y[0]
            x[1] = y[1] - 1
            return y

    periodicboundary = PeriodicBoundary()

    bcs = cashocs.create_periodic_bcs(W, boundaries, [1,2], periodicboundary)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0))
    a = inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx
    L = inner(f, v)*dx

    u = Function(W)
    cashocs._utils.assemble_and_solve_linear(a, L, u, bcs)

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

    class PeriodicBoundary(SubDomain):
        def map(self, x, y):
            x[0] = y[0]
            x[1] = y[1] - 1
            return y

    periodicboundary = PeriodicBoundary()

    inflow = Expression(("1", "0"), degree=2)
    bc0 = cashocs.create_dirichlet_bcs(W.sub(0), inflow, boundaries, 3)
    pbc = cashocs.create_periodic_bcs(W, boundaries, [1,2], periodicboundary)
    bcs = bc0 + pbc

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0))
    a = inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx
    L = inner(f, v)*dx

    u = Function(W)
    cashocs._utils.assemble_and_solve_linear(a, L, u, bcs)

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
    mesh, _, boundaries, dx, _, _ = cashocs.import_mesh("mesh/periodic_boundary_conditions/mesh_square.msh")

    V = VectorElement("CG", mesh.ufl_cell(), 2)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement([V,Q]))

    class PeriodicBoundary(SubDomain):
        def map(self, x, y):
            x[0] = y[0]
            x[1] = y[1] - 1
            return y

    periodicboundary = PeriodicBoundary()

    inflow = Expression(("1", "0"), degree=2)
    bc0 = cashocs.create_dirichlet_bcs(W.sub(0), inflow, boundaries, 1)
    bc1 = cashocs.create_dirichlet_bcs(W.sub(0), Constant((0.0, 0.0)), boundaries, 5)
    pbcs = cashocs.create_periodic_bcs(W, boundaries, [2,4], periodicboundary)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0))
    a = inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx
    L = inner(f, v)*dx

    bcs = bc0+bc1+pbcs

    u = Function(W)
    cashocs._utils.assemble_and_solve_linear(a, L, u, bcs)
    
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

    assert error < 2

def test_square_2_1():
    mesh, _, boundaries, dx, _, _ = cashocs.import_mesh("mesh/periodic_boundary_conditions/mesh_square2to1.msh")

    V = VectorElement("CG", mesh.ufl_cell(), 2)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement([V,Q]))

    class PeriodicBoundary(SubDomain):
        def map(self, x, y):
            x[0] = y[0]
            x[1] = y[1] - 1
            return y

    periodicboundary = PeriodicBoundary()

    inflow = Expression(("1", "0"), degree=2)
    bc0 = cashocs.create_dirichlet_bcs(W.sub(0), inflow, boundaries, 1)
    bc1 = cashocs.create_dirichlet_bcs(W.sub(0), Constant((0.0, 0.0)), boundaries, 5)
    pbcs = cashocs.create_periodic_bcs(W, boundaries, [2,4], periodicboundary)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0))
    a = inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx
    L = inner(f, v)*dx

    bcs = bc0+bc1+pbcs

    u = Function(W)
    cashocs._utils.assemble_and_solve_linear(a, L, u, bcs)
    
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

    assert error < 2

def test_circle_right_angle():
    mesh, _, boundaries, dx, _, _ = cashocs.import_mesh("mesh/periodic_boundary_conditions/mesh_circle90.msh")

    V = VectorElement("CG", mesh.ufl_cell(), 2)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement([V,Q]))

    class PeriodicBoundary(SubDomain):
        def map(self, x, y):
            x[0] = y[1]
            x[1] = -y[0]
            return y

    periodicboundary = PeriodicBoundary()

    inflow = Expression(("x[0]", "x[1]"), degree=2)
    bc0 = cashocs.create_dirichlet_bcs(W.sub(0), inflow, boundaries, 4)
    bc1 = cashocs.create_dirichlet_bcs(W.sub(0), Constant((0.0, 0.0)), boundaries, 5)
    pbcs = cashocs.create_periodic_bcs(W, boundaries, [1,3], periodicboundary)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0))
    a = inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx
    L = inner(f, v)*dx

    bcs = bc0+bc1+pbcs

    u = Function(W)
    cashocs._utils.assemble_and_solve_linear(a, L, u, bcs)
    
    u, p = u.split()

    x_ref = np.linspace(0.5, 1, num=50)
    coord_top = []
    coord_bottom = []

    for x in x_ref:
        value_m = u((0, x))
        coord_top.append(value_m)
        value_s = u((x, 0))
        coord_bottom.append(value_s)
    coord_top_np = np.array(coord_top)
    coord_bottom_np = np.array(coord_bottom)

    error = np.linalg.norm(coord_top_np-coord_bottom_np)

    assert error < 4

def test_circle_acute_angle():
    mesh, _, boundaries, dx, _, _ = cashocs.import_mesh("mesh/periodic_boundary_conditions/mesh_circle60.msh")

    V = VectorElement("CG", mesh.ufl_cell(), 2)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement([V,Q]))

    cosinus = np.cos(2*np.pi/5)
    sinus = np.sin(2*np.pi/5)

    class PeriodicBoundary(SubDomain):
        def map(self, x, y):
            x[0] = y[1]*sinus + y[0]*cosinus
            x[1] = -y[0]*sinus + y[1]*cosinus
            return y
        
    periodicboundary = PeriodicBoundary()

    inflow = Expression(("x[0]", "x[1]"), degree=2)
    bc0 = cashocs.create_dirichlet_bcs(W.sub(0), inflow, boundaries, 4)
    bc1 = cashocs.create_dirichlet_bcs(W.sub(0), Constant((0.0, 0.0)), boundaries, 5)
    pbcs = cashocs.create_periodic_bcs(W, boundaries, [1,3], periodicboundary)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0))
    a = inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx
    L = inner(f, v)*dx

    bcs = bc0+bc1+pbcs

    u = Function(W)
    cashocs._utils.assemble_and_solve_linear(a, L, u, bcs)
    
    u, p = u.split()

    x_ref = np.linspace(0.5, 1, num=50)
    coord_top = []
    coord_bottom = []

    for x in x_ref:
        value_m = u((cosinus*x, sinus*x))
        coord_top.append(value_m)
        value_s = u((x, 0))
        coord_bottom.append(value_s)
    coord_top_np = np.array(coord_top)
    coord_bottom_np = np.array(coord_bottom)

    error = np.linalg.norm(coord_top_np-coord_bottom_np)

    assert error < 5

def test_circle_obture_angle():
    mesh, _, boundaries, dx, _, _ = cashocs.import_mesh("mesh/periodic_boundary_conditions/mesh_circle120.msh")

    V = VectorElement("CG", mesh.ufl_cell(), 2)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement([V,Q]))

    cosinus = np.cos(2*np.pi/3)
    sinus = np.sin(2*np.pi/3)

    class PeriodicBoundary(SubDomain):
        def map(self, x, y):
            x[0] = y[1]*sinus + y[0]*cosinus
            x[1] = -y[0]*sinus + y[1]*cosinus
            return y
        
    periodicboundary = PeriodicBoundary()

    inflow = Expression(("x[0]", "x[1]"), degree=2)
    bc0 = cashocs.create_dirichlet_bcs(W.sub(0), inflow, boundaries, 4)
    bc1 = cashocs.create_dirichlet_bcs(W.sub(0), Constant((0.0, 0.0)), boundaries, 5)
    pbcs = cashocs.create_periodic_bcs(W, boundaries, [1,3], periodicboundary)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0))
    a = inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx
    L = inner(f, v)*dx

    bcs = bc0+bc1+pbcs

    u = Function(W)
    cashocs._utils.assemble_and_solve_linear(a, L, u, bcs)
    
    u, p = u.split()

    x_ref = np.linspace(0.5, 1, num=50)
    coord_top = []
    coord_bottom = []
    cosinus = np.cos(2*np.pi/3)
    sinus = np.sin(2*np.pi/3)

    for x in x_ref:
        value_m = u((cosinus*x, sinus*x))
        coord_top.append(value_m)
        value_s = u((x, 0))
        coord_bottom.append(value_s)
    coord_top_np = np.array(coord_top)
    coord_bottom_np = np.array(coord_bottom)

    error = np.linalg.norm(coord_top_np-coord_bottom_np)
    assert error < 5