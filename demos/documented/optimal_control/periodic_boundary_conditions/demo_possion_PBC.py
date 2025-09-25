from fenics import *
import cashocs
cashocs.log.set_log_level(cashocs.log.INFO)
config = cashocs.load_config("config.ini")

mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)

V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)
e = inner(grad(y), grad(p)) * dx - u * p * dx

bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [3, 4])
pbc = cashocs.create_periodic_bcs(V, boundaries, [1,2])
bc = bcs + pbc

y_d = Expression("exp(-(pow(x[0]-0.9,2)+pow(x[1]-0.5,2))/0.05)",degree=1)

alpha = 1e-6
J = cashocs.IntegralFunctional(
    Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * dx
)

ocp = cashocs.OptimalControlProblem(e, bc, J, y, u, p, config=config, control_bcs_list=pbc)
try:
    ocp.solve()
except:
    pass

import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
fig = plot(u)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Control variable u")

plt.subplot(1, 3, 2)
fig = plot(y)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("State variable y")

plt.subplot(1, 3, 3)
fig = plot(y_d, mesh=mesh)
plt.colorbar(fig, fraction=0.046, pad=0.04)
plt.title("Desired state y_d")

plt.tight_layout()
plt.show()