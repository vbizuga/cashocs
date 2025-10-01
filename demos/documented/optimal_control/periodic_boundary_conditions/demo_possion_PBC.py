# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
# ---

# ```{eval-rst}
# .. include:: ../../../global.rst
# ```
#
#
# (demo_poisson_PBC)=
# # Periodic Boundary Conditions on a Poisson Problem
#
# ## Problem Formulation
#
# Some physical contexts require the implementation of Periodic Boundary
# Conditions for more exact results. In this Demo, PBCs are applied to the "mother
# problem" of PDE constrained optimization, i.e.,
#
# $$
# \begin{align}
#     &\min\; J(y,u) = \frac{1}{2} \int_{\Omega} \left( y - y_d \right)^2
#     \text{ d}x + \frac{\alpha}{2} \int_{\Omega} u^2 \text{ d}x \\
#     &\text{ subject to } \qquad
#     \begin{alignedat}[t]{2}
#         -\Delta y &= u \quad &&\text{ in } \Omega,\\
#         y &= 0 \quad &&\text{ on } \Gamma.
#     \end{alignedat}
# \end{align}
# $$
#
# (see, e.g., [Tröltzsch - Optimal Control of Partial Differential Equations](
# https://doi.org/10.1090/gsm/112)
# or [Hinze, Pinnau, Ulbrich, and Ulbrich - Optimization with PDE constraints](
# https://doi.org/10.1007/978-1-4020-8839-1).
#
# Details on the problem under consideration are found in the Demo Poisson.
#
# ## Implementation
#
# The complete python code can be found in the file
# {download}`demo_poisson_PBC.py
# </../../demos/documented/optimal_control/periodic_boundary_conditions/demo_poisson_PBC.py>`
# and the corresponding config can be found in
# {download}`config.ini </../../demos/documented/optimal_control/periodic_boundary_conditions/config.ini>`.
#
# ### Initialization
#
# We begin by importing FEniCS and cashocs, specifing the log level of cashocs and loading the config file. 
# For the sake of better readability we use a wildcard import for FEniCS

# +

from fenics import *
import cashocs
cashocs.log.set_log_level(cashocs.log.INFO)
config = cashocs.load_config("config.ini")

# -
# In the next step, we build a quadratic mesh as in the Demo for the Poisson Problem.
# The function space and problem formulation also follows the Poisson Demo.
# +

mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)

V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)
e = inner(grad(y), grad(p)) * dx - u * p * dx

#-
# Coming to the implementation of the boundary conditions, one needs an additional function
# {py:func}`create_periodic_bcs <cashocs.create_periodic_bcs>` implementing the periodic boundary
# conditions.
# +

bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [3, 4])
pbc = cashocs.create_periodic_bcs(V, boundaries, [1,2])

# -
# This function takes the function space, the boundaries, and a list of two indices of the
# boundaries to be matched. The first index represents the master side and the second the slave side.
# Note that these two boundaries should have the same length. No spatial relation needs to be specified, 
# as rotation and translation of the DOFs to be mapped are handled internally. The function creates a 
# list of PeriodicBC objects, that can be added to the existing list of DirichletBCs:

bc = bcs + pbc

# Now this list can be passed to the solver, as seen further down.
#
# ### Definition of the cost functional
#
# To define the optimal control problem, the cost funcitonal is first specified by defining the disered state and
# the regularization parameter, i.e.,
# +

y_d = Expression("exp(-(pow(x[0]-0.9,2)+pow(x[1]-0.5,2))/0.05)",degree=1)
alpha = 1e-6
J = cashocs.IntegralFunctional(
    Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * dx
)

#-
#
# ### Definition of the optimization problem and its solution
#
# Finally, we set up an
# {py:class}`OptimalControlProblem <cashocs.OptimalControlProblem>` {python}`ocp` and
# then directly solve it with the method {py:meth}`ocp.solve()
# <cashocs.OptimalControlProblem.solve>`
#+

ocp = cashocs.OptimalControlProblem(e, bc, J, y, u, p, config=config, control_bcs_list=pbc)
try:
    ocp.solve()
except:
    pass

#-
#
# Note that the list of PBCs is passed as a control_bcs_list, as it is a pyhsical property of the mesh.
# The options for the {py:meth}`solve <cashocs.OptimalControlProblem.solve>` call can be set 
# in the config file or directly set with keyword arguments, as described in the Poisson Demo.
#
# To visualize the results we use the following code and matplotlib ::

# +
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
#-

# The output should look like this
# :::{image} /../../demos/documented/optimal_control/periodic_boundary_conditions/img_poisson_PBC.png
# :::