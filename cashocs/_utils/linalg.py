# Copyright (C) 2020-2026 Fraunhofer ITWM and Sebastian Blauth
#
# This file is part of cashocs.
#
# cashocs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cashocs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cashocs.  If not, see <https://www.gnu.org/licenses/>.

"""Linear algebra helper functions."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import fenics
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from scipy import sparse

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _exceptions
from cashocs import log
from cashocs import mpi
from cashocs._utils import forms as forms_module

if TYPE_CHECKING:
    from cashocs import _typing

iterative_ksp_options: _typing.KspOption = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_strong_threshold": 0.7,
    "ksp_rtol": 1e-20,
    "ksp_atol": 1e-50,
    "ksp_max_it": 1000,
}

direct_ksp_options: _typing.KspOption = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_24": 1,
}


from scipy.sparse import csr_matrix
from scipy.interpolate import make_interp_spline
from scipy.interpolate import CubicSpline


def split_linear_forms(forms: list[ufl.Form]) -> tuple[list[ufl.Form], list[ufl.Form]]:
    """Splits a list of linear forms into left- and right-hand sides.

    Args:
        forms: A list of (linear) ufl forms.

    Returns:
        A tuple (lhs_forms, rhs_forms), where lhs_forms is the list of forms of the
        left-hand sides, and rhs_forms is the list of forms of the right-hand side.

    """
    lhs_list = []
    rhs_list = []
    for i in range(len(forms)):
        try:
            lhs, rhs = ufl.system(forms[i])
        except ufl.log.UFLException as ufl_exception:
            raise _exceptions.CashocsException(
                "The state system could not be transferred to a linear "
                "system.\n"
                "Perhaps you specified that the system is linear, "
                "although it is not.\n"
                "In your config, in the StateSystem section, "
                "try using is_linear = False."
            ) from ufl_exception
        lhs_list.append(lhs)

        if rhs.empty():
            test_function = lhs.arguments()[0]
            mesh = lhs.ufl_domain()
            dx = ufl.Measure("dx", mesh)
            zero_form = (
                ufl.dot(
                    fenics.Constant(np.zeros(test_function.ufl_shape)),
                    test_function,
                )
                * dx
            )
            rhs_list.append(zero_form)
        else:
            rhs_list.append(rhs)

    return lhs_list, rhs_list


@log.profile_execution_time("assembling the linear system")
def assemble_petsc_system(
    lhs_form: ufl.Form,
    rhs_form: ufl.Form,
    bcs: fenics.DirichletBC | forms_module.PerdiodicBC | list[fenics.DirichletBC|forms_module.PerdiodicBC] | None = None,
    A_tensor: fenics.PETScMatrix | None = None,  # pylint: disable=invalid-name
    b_tensor: fenics.PETScVector | None = None,
    preconditioner_form: ufl.Form | None = None,
    comm: MPI.Comm | None = None,
) -> tuple[PETSc.Mat, PETSc.Vec, PETSc.Mat]:
    """Assembles a system symmetrically and converts objects to PETSc format.

    Args:
        lhs_form: The UFL form for the left-hand side of the linear equation.
        rhs_form: The UFL form for the right-hand side of the linear equation.
        bcs: A list of Dirichlet boundary conditions or Periodic boundary conditions.
        A_tensor: A matrix into which the result is assembled. Default is ``None``.
        b_tensor: A vector into which the result is assembled. Default is ``None``.
        preconditioner_form: The UFL form for assembling the preconditioner. Must
            be a bilinear form.
        comm: The MPI communicator for the problem.

    Returns:
        A tuple (A, b), where A is the matrix of the linear system, and b is the vector
        of the linear system.

    Notes:
        This function always uses the ident_zeros method of the matrix in order to add a
        one to the diagonal in case the corresponding row only consists of zeros. This
        allows for well-posed problems on the boundary etc.

    """
    if comm is None:
        comm = mpi.COMM_WORLD

    mod_lhs_form = forms_module.bilinear_boundary_form_modification([lhs_form])[0]
    if A_tensor is None:
        A_tensor = fenics.PETScMatrix(comm)
    if b_tensor is None:
        b_tensor = fenics.PETScVector(comm)

    # Split boundary conditions
    dbcs = []
    pbcs = []
    for element in bcs:
        if type(element) == fenics.DirichletBC:
            dbcs += [element]
        elif type(element) == forms_module.PeriodicBC:
            pbcs += [element]
        else:
            raise _exceptions.CashocsException(
                "Boundary conditions must match one of the"
                "following types:\n"
                "fenics.DirichletBC\n"
                "cashocs._utils._forms.PeriodicBC"
            )

    try:
        fenics.assemble_system(
            mod_lhs_form,
            rhs_form,
            dbcs,
            keep_diagonal=True,
            A_tensor=A_tensor,
            b_tensor=b_tensor,
        )

    except ValueError as value_exception:
        raise _exceptions.CashocsException(
            "The state system could not be transferred to a linear "
            "system.\n"
            "Perhaps you specified that the system is linear, "
            "although it is not.\n"
            "In your config, in the StateSystem section, "
            "try using is_linear = False."
        ) from value_exception
    A_tensor.ident_zeros()

    if preconditioner_form is not None:
        P_tensor = fenics.PETScMatrix()  # pylint: disable=invalid-name
        c_tensor = fenics.PETScVector()
        fenics.assemble_system(
            preconditioner_form,
            rhs_form,
            dbcs,
            keep_diagonal=True,
            A_tensor=P_tensor,
            b_tensor=c_tensor,
        )
        P_tensor.ident_zeros()
        P = P_tensor.mat()  # pylint: disable=invalid-name
    else:
        P = None  # pylint: disable=invalid-name

    for pbc in pbcs:
        PBCI = PeriodicBoundaryInterpolator(pbc.functionspace, pbc.boundaries, A_tensor, b_tensor)
        A, b = PBCI.assemble_periodic_system(pbc.master_idc,pbc.slave_idc)

    A = A_tensor.mat()  # pylint: disable=invalid-name
    b = b_tensor.vec()

    log.end()

    return A, b, P


def setup_petsc_options(
    objs: list[PETSc.KSP | PETSc.SNES], ksp_options: list[_typing.KspOption]
) -> None:
    """Sets up an (iterative) linear solver.

    This is used to pass user defined command line type options for PETSc
    to the PETSc KSP objects. Here, options[i] is applied to ksps[i].

    Args:
        objs: A list of PETSc objects (e.g. linear solvers) to which the (command line)
            options are applied to.
        ksp_options: A list of command line options that specify the solver
            from PETSc.

    """
    fenics.PETScOptions.clear()
    opts = PETSc.Options()

    for i in range(len(objs)):
        opts.clear()

        log.trace(f"Setting PETSc options for {objs[i]}")
        for key, value in ksp_options[i].items():
            log.trace(f"{key}: {value}")
            opts.setValue(key, value)

        objs[i].setFromOptions()


def _set_fieldsplit_index_sets(
    pc: PETSc.PC, function_space: fenics.FunctionSpace
) -> None:
    idx = []
    name = []
    for i in range(function_space.num_sub_spaces()):
        idx_i = PETSc.IS().createGeneral(function_space.sub(i).dofmap().dofs())
        idx.append(idx_i)
        name.append(f"{i:d}")

    idx_tuples = list(zip(name, idx, strict=True))
    try:
        pc.setFieldSplitIS(*idx_tuples)
    finally:
        for idx_i in idx:
            idx_i.destroy()


def setup_fieldsplit_preconditioner(
    fun: fenics.Function | None,
    ksp: PETSc.KSP,
    options: _typing.KspOption,
) -> None:
    """Sets up the preconditioner for the fieldsplit case.

    This defines the index sets which indicate where the splitting should take place.

    Args:
        fun: The function corresponding to the mixed system to be solved.
        ksp: The ksp object.
        options: The options for the ksp.

    """
    if fun is not None:
        if "pc_type" in options.keys() and options["pc_type"] == "fieldsplit":
            function_space = fun.function_space()
            comm = function_space.mesh().mpi_comm()
            if not function_space.num_sub_spaces() > 1:
                raise _exceptions.InputError(
                    "cashocs._utils.solve_linear_problem",
                    "ksp_options",
                    "You have specified a fieldsplit preconditioner, but the "
                    "problem to be solved is not a mixed one.",
                )

            if not any(key.endswith("_fields") for key in options.keys()):
                pc = ksp.getPC()
                pc.setType(PETSc.PC.Type.FIELDSPLIT)
                _set_fieldsplit_index_sets(pc, function_space)
            else:
                dof_total = function_space.dofmap().dofs()
                offset = np.min(dof_total)

                num_sub_spaces = function_space.num_sub_spaces()
                dof_list = [
                    np.array(function_space.sub(i).dofmap().dofs())
                    for i in range(num_sub_spaces)
                ]

                section = PETSc.Section().create(comm)
                section.setNumFields(num_sub_spaces)

                for i in range(num_sub_spaces):
                    section.setFieldName(i, f"{i:d}")
                    section.setFieldComponents(i, 1)
                section.setChart(0, len(dof_total))
                for field_idx, dofs in enumerate(dof_list):
                    for i in dofs:
                        section.setDof(i - offset, 1)
                        section.setFieldDof(i - offset, field_idx, 1)
                section.setUp()

                dm = PETSc.DMShell().create(comm)
                dm.setDefaultSection(section)
                dm.setUp()

                ksp.setDM(dm)
                petsc_version = PETSc.Sys.getVersion()
                if petsc_version[0] == 3 and petsc_version[1] >= 25:
                    ksp.setDMActive(ksp.DMActive.ALL, False)
                else:
                    ksp.setDMActive(False)
                dm.destroy()
                section.destroy()


def define_ksp_options(
    ksp_options: _typing.KspOption | None = None,
) -> _typing.KspOption:
    """Defines the KSP options to be used by PETSc.

    If no options are supplied, the direct solver mumps will be used.

    Args:
        ksp_options: The KSP options for PETSc

    Returns:
        The KSP options that are supplied of the default ones (for mumps).

    """
    if ksp_options is None:
        options = copy.deepcopy(direct_ksp_options)
    else:
        options = ksp_options

    return options


def setup_matrix_and_preconditioner(
    ksp: PETSc.KSP,
    A: PETSc.Mat | None = None,  # pylint: disable=invalid-name
    P: PETSc.Mat | None = None,  # pylint: disable=invalid-name
) -> None:
    """Set up the system matrix and preconditioner for a linear solve.

    Args:
        ksp: The KSP object used to solve the problem.
        A: The system matrix or `None`.
        P: The preconditioner matrix or `None`.

    """
    if A is not None:
        if P is None:
            ksp.setOperators(A)
        else:
            ksp.setOperators(A, P)
    else:
        A = ksp.getOperators()[0]
        if A.size[0] == -1 and A.size[1] == -1:
            raise _exceptions.InputError(
                "cashocs._utils.solve_linear_problem",
                "ksp",
                "The KSP object has to be initialized with some Matrix in case A is "
                "None.",
            )


def solve_linear_problem(
    A: PETSc.Mat | None = None,  # pylint: disable=invalid-name
    b: PETSc.Vec | None = None,
    fun: fenics.Function | None = None,
    ksp_options: _typing.KspOption | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    P: PETSc.Mat | None = None,  # pylint: disable=invalid-name
    linear_solver: LinearSolver | None = None,
) -> PETSc.Vec:
    """Solves a finite dimensional linear problem.

    An overview over possible command line options for the PETSc KSP object can
    be found at `<https://petsc.org/release/manualpages/KSP/>`_ and options for the
    preconditioners can be found at `<https://petsc.org/release/manualpages/PC/>`_.

    Args:
        A: The PETSc matrix corresponding to the left-hand side of the problem. If
            this is None, then the matrix stored in the ksp object is used. Raises
            an error if no matrix is stored. Default is None.
        b: The PETSc vector corresponding to the right-hand side of the problem.
            If this is None, then a zero right-hand side is assumed, and a zero vector
            is returned. Default is None.
        fun: The function which will store the solution of the problem. If this is
            None, then a new vector will be created (and returned).
        ksp_options: The options for the PETSc ksp object. If this is None (the default)
            a direct method is used.
        rtol: The relative tolerance used in case an iterative solver is used for
            solving the linear problem. Overrides the specification in the ksp object
            and ksp_options.
        atol: The absolute tolerance used in case an iterative solver is used for
            solving the linear problem. Overrides the specification in the ksp object
            and ksp_options.
        comm: The MPI communicator for the problem.
        P: The PETSc matrix corresponding to the preconditioner. If this is None, then
            the left-hand-side matrix A is used to build the preconditioner.
        linear_solver: The LinearSolver that should be used to solve the problem.

    Returns:
        The solution vector.

    """
    log.warning(
        "The function cashocs._utils.linalg.solve_linear_problem is "
        "deprecated and will be removed in a future version. Please use"
        "the solve method of cashocs._utils.linalg.LinearSolver instead."
    )

    if linear_solver is None:
        linear_solver = LinearSolver()

    return linear_solver.solve(
        fun, A=A, b=b, ksp_options=ksp_options, rtol=rtol, atol=atol, P=P
    )


def assemble_and_solve_linear(
    lhs_form: ufl.Form,
    rhs_form: ufl.Form,
    function: fenics.Function,
    bcs: fenics.DirichletBC | list[fenics.DirichletBC] | None = None,
    A: fenics.PETScMatrix | None = None,  # pylint: disable=invalid-name
    b: fenics.PETScVector | None = None,
    ksp_options: _typing.KspOption | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    preconditioner_form: ufl.Form | None = None,
    linear_solver: LinearSolver | None = None,
) -> None:
    """Assembles and solves a linear system.

    Note that the solution is stored in the fenics.Function `function`, which is
    a required input parameter to this function.

    Args:
        lhs_form: The UFL form for the left-hand side of the linear equation.
        rhs_form: The UFL form for the right-hand side of the linear equation.
        function: The function which will be solution of the problem.
        bcs: A list of Dirichlet boundary conditions.
        A: A matrix into which the lhs is assembled. Default is ``None``.
        b: A vector into which the rhs is assembled. Default is ``None``.
        ksp_options: The options for the PETSc ksp object. If this is None (the default)
            a direct method is used.
        rtol: The relative tolerance used in case an iterative solver is used for
            solving the linear problem. Overrides the specification in the ksp object
            and ksp_options.
        atol: The absolute tolerance used in case an iterative solver is used for
            solving the linear problem. Overrides the specification in the ksp object
            and ksp_options.
        preconditioner_form: The UFL for assembling the preconditioner. Must be a
            bilinear form.
        linear_solver: The LinearSolver used to solve the problem.

    Returns:
        A PETSc vector containing the solution x.

    """
    # pylint: disable=invalid-name
    comm = function.function_space().mesh().mpi_comm()
    A_matrix, b_vector, P_matrix = assemble_petsc_system(
        lhs_form,
        rhs_form,
        bcs,
        A_tensor=A,
        b_tensor=b,
        preconditioner_form=preconditioner_form,
        comm=comm,
    )

    if linear_solver is None:
        linear_solver = LinearSolver()

    linear_solver.solve(
        function,
        A=A_matrix,
        b=b_vector,
        ksp_options=ksp_options,
        rtol=rtol,
        atol=atol,
        P=P_matrix,
    )


class LinearSolver:
    """A solver for linear problems arising from discretized PDEs."""

    @log.profile_execution_time("solving the linear system with PETSc KSP")
    def solve(
        self,
        function: fenics.Function,
        A: PETSc.Mat | None = None,  # pylint: disable=invalid-name
        b: PETSc.Vec | None = None,
        ksp_options: _typing.KspOption | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        P: PETSc.Mat | None = None,  # pylint: disable=invalid-name
    ) -> None:
        """Solves a finite dimensional linear problem arising from a discretized PDE.

        Args:
            function: The function which will store the solution of the problem.
            A: The PETSc matrix corresponding to the left-hand side of the problem. If
                this is None, then the matrix stored in the ksp object is used. Raises
                an error if no matrix is stored. Default is None.
            b: The PETSc vector corresponding to the right-hand side of the problem.
                If this is None, then a zero right-hand side is assumed, and a zero
                vector is returned. Default is None.
            ksp_options: The options for the PETSc ksp object. If this is None (the
                default) a direct method is used.
            rtol: The relative tolerance used in case an iterative solver is used for
                solving the linear problem. Overrides the specification in the ksp
                object and ksp_options.
            atol: The absolute tolerance used in case an iterative solver is used for
                solving the linear problem. Overrides the specification in the ksp
                object and ksp_options.
            P: The PETSc matrix corresponding to the preconditioner.

        Returns:
            The solution vector.

        """
        self.comm = function.function_space().mesh().mpi_comm()
        ksp = PETSc.KSP().create(self.comm)
        setup_matrix_and_preconditioner(ksp, A, P)

        if b is None:
            function.vector().vec().set(0.0)
            function.vector().apply("")

        x = function.vector().vec()

        options = define_ksp_options(ksp_options)

        setup_fieldsplit_preconditioner(function, ksp, options)
        setup_petsc_options([ksp], [options])

        ksp.setTolerances(rtol=rtol, atol=atol)

        try:
            ksp.solve(b, x)
            converged_reason = ksp.getConvergedReason()
        finally:
            ksp.destroy()
            if hasattr(PETSc, "garbage_cleanup"):
                PETSc.garbage_cleanup(comm=self.comm)

        if converged_reason < 0:
            raise _exceptions.PETScKSPError(converged_reason)

        function.vector().apply("")


class EquationResidualComputer:
    """Computes equation-wise residual norms for a fixed function space."""

    def __init__(self, function_space: fenics.FunctionSpace) -> None:
        """Initializes the residual computer.

        Args:
            function_space: The function space defining the equations.

        """
        self.function_space = function_space
        if isinstance(
            function_space.ufl_element(), ufl.VectorElement | ufl.FiniteElement
        ):
            self.is_sets = []
        else:
            self.is_sets = [
                PETSc.IS().createGeneral(function_space.sub(i).dofmap().dofs())
                for i in range(function_space.num_sub_spaces())
            ]

    def compute(self, residual: PETSc.Vec) -> np.ndarray:
        """Computes equation-wise residual norms.

        Args:
            residual: The PETSc vector containing the (nonlinear) residual.

        Returns:
            A list of residual norms for the individual equations.

        """
        equation_residual_vectors = []
        try:
            for iset in self.is_sets:
                equation_residual_vectors.append(residual.getSubVector(iset))

            return np.array(
                [r.norm(PETSc.NormType.NORM_2) for r in equation_residual_vectors]
            )
        finally:
            for iset, vector in zip(
                self.is_sets, equation_residual_vectors, strict=False
            ):
                residual.restoreSubVector(iset, vector)

    def destroy(self) -> None:
        """Destroys the cached PETSc index sets."""
        for iset in self.is_sets:
            iset.destroy()
        self.is_sets = []


def compute_equation_residuals(
    residual: PETSc.Vec, function_space: fenics.FunctionSpace
) -> np.ndarray:
    """Splits the residual to the individual equations and computes the norm.

    Args:
        residual: The PETSc vector containing the (nonlinear) residual.
        function_space: The function space defining the equations.

    Returns:
        A list of residual norms for the individual equations.

    """
    residual_computer = EquationResidualComputer(function_space)
    try:
        equation_residual_norms = residual_computer.compute(residual)
    finally:
        residual_computer.destroy()

    return equation_residual_norms


class Interpolator:
    """Efficient interpolation between two function spaces.

    This is very useful, if multiple interpolations have to be carried out between the
    same spaces, which is made significantly faster by computing the corresponding
    matrix. The function spaces can even be defined on different meshes.

    Notes:
        This class only works properly for continuous Lagrange elements and constant,
        discontinuous Lagrange elements.

    Examples:
        Here, we consider interpolating from CG1 elements to CG2 elements ::

            import fenics
            import cashocs

            mesh, _, _, _, _, _ = cashocs.regular_mesh(25)
            V1 = fenics.FunctionSpace(mesh, 'CG', 1)
            V2 = fenics.FunctionSpace(mesh, 'CG', 2)

            expr = fenics.Expression('sin(2*pi*x[0])', degree=1)
            u = fenics.interpolate(expr, V1)

            interp = cashocs._utils.Interpolator(V1, V2)
            interp.interpolate(u)

    """

    def __init__(
        self, origin_space: fenics.FunctionSpace, target_space: fenics.FunctionSpace
    ) -> None:
        """Initializes self.

        Args:
            origin_space: The function space whose objects shall be interpolated.
            target_space: The space into which they shall be interpolated.

        """
        if not (
            origin_space.ufl_element().family() == "Lagrange"
            or (
                origin_space.ufl_element().family() == "Discontinuous Lagrange"
                and origin_space.ufl_element().degree() == 0
            )
        ):
            raise _exceptions.InputError(
                "cashocs._utils.Interpolator",
                "origin_space",
                "The interpolator only works with CG n or DG 0 elements",
            )
        if not (
            target_space.ufl_element().family() == "Lagrange"
            or (
                target_space.ufl_element().family() == "Discontinuous Lagrange"
                and target_space.ufl_element().degree() == 0
            )
        ):
            raise _exceptions.InputError(
                "cashocs._utils.Interpolator",
                "target_space",
                "The interpolator only works with CG n or DG 0 elements",
            )

        self.origin_space = origin_space
        self.target_space = target_space
        self.transfer_matrix = fenics.PETScDMCollection.create_transfer_matrix(
            self.origin_space, self.target_space
        ).mat()

    def interpolate(self, u: fenics.Function) -> fenics.Function:
        """Interpolates function to target space.

        The function has to belong to the origin space, i.e., the first argument
        of __init__, and it is interpolated to the destination space, i.e., the
        second argument of __init__. There is no need to call set_allow_extrapolation
        on the function (this is done automatically due to the method).

        Args:
            u: The function that shall be interpolated.

        Returns:
            The result of the interpolation.

        """
        v = fenics.Function(self.target_space)
        x = fenics.as_backend_type(u.vector()).vec()
        self.transfer_matrix.mult(x, v.vector().vec())
        v.vector().apply("")

        return v


def sparse2scipy(
    csr: tuple[np.ndarray, np.ndarray, np.ndarray], shape: tuple[int, int] | None = None
) -> sparse.csr_matrix:
    """Converts a sparse matrix representation to a sparse scipy matrix.

    Args:
        csr: The tuple making up the CSR matrix: `rows, cols, vals`.
        shape: The shape of the sparse matrix.

    Returns:
        The corresponding sparse scipy csr matrix.

    """
    rows = csr[0]
    cols = csr[1]
    vals = csr[2]
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=shape)
    return matrix


def scipy2petsc(
    scipy_matrix: sparse.csr_matrix,
    comm: MPI.Comm,
    local_size: int | None = None,
) -> PETSc.Mat:
    """Converts a sparse scipy matrix to a (sparse) PETSc matrix.

    Args:
        scipy_matrix: The sparse scipy matrix
        comm: The MPI communicator used for distributing the mesh
        local_size: The local size (number of rows) of the matrix, different for
            each process. If this is `None` (the default), then PETSc.DECIDE is used.

    Returns:
        The corresponding sparse PETSc matrix.

    """
    shape = scipy_matrix.shape

    no_rows_total = comm.allreduce(shape[0], op=MPI.SUM)
    if local_size is None:
        local_size = PETSc.DECIDE

    petsc_matrix = PETSc.Mat().createAIJ(
        comm=comm,
        size=((shape[0], no_rows_total), (local_size, shape[1])),
        csr=(scipy_matrix.indptr, scipy_matrix.indices, scipy_matrix.data),
    )

    return petsc_matrix


def l2_projection(
    expr: ufl.core.expr.Expr,
    function_space: fenics.FunctionSpace,
    ksp_options: _typing.KspOption | None = None,
    measure: ufl.Measure | None = None,
) -> fenics.Function:
    """Computes an L2 projection of an expression into the specified function space.

    Args:
        expr (ufl.core.expr.Expr): The expression that shall be projected.
        function_space (fenics.FunctionSpace): The function space into which the
            projection takes place.
        ksp_options (_typing.KspOption | None, optional): The options for the solution
            of the linear problem. If this is None, a CG method with AMG is used.
            Defaults to None.
        measure: The ufl.Measure where the L2 projection should take place.

    Returns:
        The result of the projection.

    """
    if measure is None:
        mesh = function_space.mesh()
        dx = ufl.Measure("dx", domain=mesh)
    else:
        dx = measure

    res = fenics.Function(function_space)
    u = fenics.TrialFunction(function_space)
    v = fenics.TestFunction(function_space)

    lhs = ufl.inner(u, v) * dx
    rhs = ufl.inner(expr, v) * dx

    bcs: list[fenics.DirichletBC] = []

    if ksp_options is None:
        ksp_options = copy.deepcopy(iterative_ksp_options)

    assemble_and_solve_linear(lhs, rhs, res, bcs=bcs, ksp_options=ksp_options)

    return res


class PeriodicBoundaryInterpolator:
    """Interpolation between two boundaries of the function space and assembling
    the corresponding matrix and vector.

    Args:
        function_space: The function space onto which the BCs should be imposed on.
        boundaries: The :py:class:`fenics.MeshFunction` object representing the
            boundaries.
        matrix: The fenics.PETScMatrix object on which the periodic BCs should be applied.
        vector: The fenics.PETScVector object on which the periodic BCs should be applied.

    Returns:
        A PeriodicBoundaryInterpolator class that is later supposed to be used by
        calling the subfunction assemble_periodic_system to obtain the assembled 
        matrix and vector.

    Notes:
        This class only works properly for linear and quadratic function spaces.
        The supported types of function spaces are:
        FiniteElement,
        VectorElement,
        MixedElement of FiniteElement or VectorElement.

    """
    def __init__(self, functionspace, boundaries, matrix, vector):
        self.boundaries = boundaries
        self.functionspace : fenics.FunctionSpace = functionspace
        self.mesh = self.functionspace.mesh()
        self.matrix = fenics.as_backend_type(matrix).mat()
        self.matrix_scipy = csr_matrix(self.matrix.getValuesCSR()[::-1],shape=self.matrix.size)
        self.vector = fenics.as_backend_type(vector).vec()
        self.dof_coord = self.functionspace.tabulate_dof_coordinates()
        self.rotation : float = 0.0
        self.sorted_indices : list = []

    def get_interp_type(self, functionspace):
        if functionspace.ufl_element().degree() == 1:
            interp_type = 'linear'
        elif functionspace.ufl_element().degree() == 2:
            interp_type = 'quadratic'
        elif functionspace.ufl_element().degree() == 3:
            interp_type = 'cubic'
        else:
            interp_type = 'quadratic'
        
        return interp_type

    def create_interpolation_matrix(self, src, dst, interp_type):
        n_src = len(src)
        n_dst = len(dst)
        interpolation_matrix = np.zeros((n_dst, n_src))
        basis = np.identity(n_src)

        if interp_type == 'linear':
            for i in range(n_src):
                interpolation_matrix[:, i] = np.interp(dst, src, basis[i])

        elif interp_type == 'quadratic':
            for i in range(n_src):
                interpolation_matrix[:, i] = make_interp_spline(src, basis[i], k=2)(dst)

        elif interp_type == 'cubic':
            for i in range(n_src):
                interpolation_matrix[:, i] = CubicSpline(src, basis[i], bc_type='natural')(dst)

        else:
            raise RuntimeError("false interpolation type")

        return interpolation_matrix

    def matrix_manipulation_scalar(self, dofs_m, dofs_s, int_matrix):
        # Slave Zeilen und Spalten werden auf die entprechende Master Zeile und Spalte addiert
        self.matrix_scipy[:,dofs_m] += self.matrix_scipy[:,dofs_s] @ int_matrix
        self.matrix_scipy[dofs_m,:] += int_matrix.T @ self.matrix_scipy[dofs_s,:]
        self.vector[dofs_m] += int_matrix.T @ self.vector[dofs_s]

        # Slave Zeilen und Spalten nullen
        self.matrix_scipy[dofs_s,:] = 0
        self.matrix_scipy[:,dofs_s] = 0

        # Anpassen der Slave Zeilen und des Lösungsvektors
        for index_s, dof_s in enumerate(dofs_s):
            self.vector[dof_s] = 0
            self.matrix_scipy[dof_s,dof_s] = 1
            self.matrix_scipy[dof_s,dofs_m] = -int_matrix[index_s,:]

    def matrix_manipulation_vector(self, dofs_m_x, dofs_s_x, dofs_m_y, dofs_s_y, int_matrix):
        # Berechne Winkel. Annahme: Strecken ohne Knick, Dreiecksberechnung
        vec_m = [self.dof_coord[dofs_m_x[0]][0]-self.dof_coord[dofs_m_x[-1]][0],self.dof_coord[dofs_m_x[0]][1]-self.dof_coord[dofs_m_x[-1]][1]]
        vec_s = [self.dof_coord[dofs_s_x[0]][0]-self.dof_coord[dofs_s_x[-1]][0],self.dof_coord[dofs_s_x[0]][1]-self.dof_coord[dofs_s_x[-1]][1]]
        
        self.rotation = np.arccos(np.round((vec_m[0]*vec_s[0]+vec_m[1]*vec_s[1])/(vec_m[0]**2+vec_m[1]**2)**(1/2)/(vec_s[0]**2+vec_s[1]**2)**(1/2),6))
        theta = np.arccos((vec_s[0])/(vec_s[0]**2+vec_s[1]**2)**(1/2))
        phi = np.arccos((vec_m[0])/(vec_m[0]**2+vec_m[1]**2)**(1/2))

        if theta - phi > fenics.DOLFIN_EPS:
            self.rotation = - self.rotation

        sinus = np.round(np.sin(self.rotation), 6)
        cosinus = np.round(np.cos(self.rotation), 6)

        # Slave Zeilen und Spalten werden auf die entprechende Master Zeile und Spalte addiert
        self.matrix_scipy[:,dofs_m_x] += self.matrix_scipy[:,dofs_s_x] @ int_matrix * cosinus + self.matrix_scipy[:,dofs_s_y] @ int_matrix * sinus
        self.matrix_scipy[dofs_m_x,:] += int_matrix.T @ self.matrix_scipy[dofs_s_x,:] * cosinus + int_matrix.T @ self.matrix_scipy[dofs_s_y,:]  * sinus
        self.matrix_scipy[:,dofs_m_y] += self.matrix_scipy[:,dofs_s_y] @ int_matrix * cosinus - self.matrix_scipy[:,dofs_s_x] @ int_matrix * sinus
        self.matrix_scipy[dofs_m_y,:] += int_matrix.T @ self.matrix_scipy[dofs_s_y,:]  * cosinus - int_matrix.T @ self.matrix_scipy[dofs_s_x,:] * sinus
        self.vector[dofs_m_x] += int_matrix.T @ self.vector[dofs_s_x] * cosinus + int_matrix.T @ self.vector[dofs_s_y]  * sinus
        self.vector[dofs_m_y] += int_matrix.T @ self.vector[dofs_s_y]  * cosinus - int_matrix.T @ self.vector[dofs_s_x] * sinus

        # Slave Zeilen und Spalten nullen
        self.matrix_scipy[dofs_s_x,:] = 0
        self.matrix_scipy[:,dofs_s_x] = 0
        self.matrix_scipy[dofs_s_y,:] = 0
        self.matrix_scipy[:,dofs_s_y] = 0
        
        for index_s_x, dof_s_x in enumerate(dofs_s_x):
            self.vector[dof_s_x] = 0
            self.matrix_scipy[dof_s_x,dof_s_x] = 1
            self.matrix_scipy[dof_s_x,dofs_m_x] = - int_matrix[index_s_x,:]*cosinus
            self.matrix_scipy[dof_s_x,dofs_m_y] = + int_matrix[index_s_x,:]*sinus

        for index_s_y, dof_s_y in enumerate(dofs_s_y):
            self.vector[dof_s_y] = 0
            self.matrix_scipy[dof_s_y,dof_s_y] = 1
            self.matrix_scipy[dof_s_y,dofs_m_y] = - int_matrix[index_s_y,:]*cosinus
            self.matrix_scipy[dof_s_y,dofs_m_x] = - int_matrix[index_s_y,:]*sinus

    def assemble_periodic_system(self, boundary_ind_master, boundary_ind_slave):
        noslipx = fenics.Constant((-5.0, 0.0))
        noslipy = fenics.Constant((0.0, -5.0))
        noslip = fenics.Constant(-5.0)
        aux_bcs = []
        
        # Create auxiliary Dirichleit Conditions to obtain dof coordinates
        if self.functionspace.num_sub_spaces() == 0:
            aux_bcs += forms_module.create_dirichlet_bcs(self.functionspace, noslip, self.boundaries, boundary_ind_master)
            aux_bcs += forms_module.create_dirichlet_bcs(self.functionspace, noslip, self.boundaries, boundary_ind_slave)

        elif isinstance(self.functionspace.ufl_element(), fenics.VectorElement):
            aux_bcs += forms_module.create_dirichlet_bcs(self.functionspace, noslipx, self.boundaries, boundary_ind_master)
            aux_bcs += forms_module.create_dirichlet_bcs(self.functionspace, noslipx, self.boundaries, boundary_ind_slave)
            aux_bcs += forms_module.create_dirichlet_bcs(self.functionspace, noslipy, self.boundaries, boundary_ind_master)
            aux_bcs += forms_module.create_dirichlet_bcs(self.functionspace, noslipy, self.boundaries, boundary_ind_slave)

        elif isinstance(self.functionspace.ufl_element(), fenics.MixedElement):
            for subspace_index in range(self.functionspace.num_sub_spaces()):
                if self.functionspace.sub(subspace_index).num_sub_spaces() == 0:
                    aux_bcs += forms_module.create_dirichlet_bcs(self.functionspace.sub(subspace_index), noslip, self.boundaries, boundary_ind_master)
                    aux_bcs += forms_module.create_dirichlet_bcs(self.functionspace.sub(subspace_index), noslip, self.boundaries, boundary_ind_slave)
                elif isinstance(self.functionspace.sub(subspace_index).ufl_element(), fenics.VectorElement):
                    aux_bcs += forms_module.create_dirichlet_bcs(self.functionspace.sub(subspace_index), noslipx, self.boundaries, boundary_ind_master)
                    aux_bcs += forms_module.create_dirichlet_bcs(self.functionspace.sub(subspace_index), noslipx, self.boundaries, boundary_ind_slave)
                    aux_bcs += forms_module.create_dirichlet_bcs(self.functionspace.sub(subspace_index), noslipy, self.boundaries, boundary_ind_master)
                    aux_bcs += forms_module.create_dirichlet_bcs(self.functionspace.sub(subspace_index), noslipy, self.boundaries, boundary_ind_slave)
                else:
                    print(self.functionspace.sub(subspace_index).num_sub_spaces())
                    raise RuntimeError("subspace dimension not fitting")
                
        else:
            raise RuntimeError("function spaces not fitting")

        # Find dof coordinates for velocity and pressure with dummy functions and sort them
        for i, bc in enumerate(aux_bcs):
            t = fenics.Function(self.functionspace)
            tv = t.vector()
            bc.apply(tv)
            structured_indices = np.where(abs(np.array(tv)+5.0) < fenics.DOLFIN_EPS_LARGE)[0]
            self.sorted_indices += [structured_indices[np.argsort(abs(self.dof_coord[:,1][structured_indices])+abs(self.dof_coord[:,0][structured_indices]))].astype('int32')]
            
        # Matrix Manipulation
        if self.functionspace.num_sub_spaces() == 0:
            interp_type = self.get_interp_type(self.functionspace)
            dofs_node_btwn_coordinates = [[0],[0]]

            # Get the relative distances of the dofs (starting at 0), to get a 1d array for the interpolation matrix
            for i in range(2):
                for index, dof in enumerate(self.sorted_indices[i]):
                    if index < len(self.sorted_indices[i])-1:
                        dofs_node_btwn_coordinates[i] += [((self.dof_coord[dof][1] - self.dof_coord[self.sorted_indices[i][index+1]][1])**2
                                                        +(self.dof_coord[dof][0] - self.dof_coord[self.sorted_indices[i][index+1]][0])**2)**(1/2) 
                                                        + dofs_node_btwn_coordinates[i][index]]
            
            # create interpolation matrix and manipulate the scipy matrix
            int_matrix_node = self.create_interpolation_matrix(dofs_node_btwn_coordinates[0],dofs_node_btwn_coordinates[1],interp_type)
            self.matrix_manipulation_scalar(self.sorted_indices[0], self.sorted_indices[1], int_matrix_node)

        elif isinstance(self.functionspace.ufl_element(), fenics.VectorElement):
            interp_type = self.get_interp_type(self.functionspace)

            if interp_type == 'quadratic' or interp_type == 'cubic':
                dofs_node_btwn = [[],[],[],[],[],[],[],[]] # master x, slave x, master y, slave y *2 node, side
                dofs_node_btwn_coordinates = [[0],[0],[0],[0]]

                for i in range(4):
                    for index, dof in enumerate(self.sorted_indices[i]):
                        if index%2 == 0:
                            dofs_node_btwn[i*2] += [dof]
                        else:
                            dofs_node_btwn[i*2+1] += [dof]
                
                for i in range(4):
                    for index, dof in enumerate(dofs_node_btwn[i]):
                        if index < len(dofs_node_btwn[i])-1:
                            dofs_node_btwn_coordinates[i] += [((self.dof_coord[dof][1] - self.dof_coord[dofs_node_btwn[i][index+1]][1])**2
                                                            +(self.dof_coord[dof][0] - self.dof_coord[dofs_node_btwn[i][index+1]][0])**2)**(1/2) 
                                                            + dofs_node_btwn_coordinates[i][index]]

                int_matrix_node = self.create_interpolation_matrix(dofs_node_btwn_coordinates[0],dofs_node_btwn_coordinates[2],interp_type)
                int_matrix_btwn = self.create_interpolation_matrix(dofs_node_btwn_coordinates[1],dofs_node_btwn_coordinates[3],interp_type)
                self.matrix_manipulation_vector(dofs_node_btwn[0][1:], dofs_node_btwn[2][1:], dofs_node_btwn[4][1:], dofs_node_btwn[6][1:], int_matrix_node[1:,1:])
                self.matrix_manipulation_vector(dofs_node_btwn[1], dofs_node_btwn[3], dofs_node_btwn[5], dofs_node_btwn[7], int_matrix_btwn)
            
            elif interp_type == 'linear':
                dofs_node_btwn_coordinates = [[0],[0]]

                for i in range(2):
                    for index, dof in enumerate(self.sorted_indices[i]):
                        if index < len(self.sorted_indices[i])-1:
                            dofs_node_btwn_coordinates[i] += [((self.dof_coord[dof][1] - self.dof_coord[self.sorted_indices[i][index+1]][1])**2
                                                            +(self.dof_coord[dof][0] - self.dof_coord[self.sorted_indices[i][index+1]][0])**2)**(1/2) 
                                                            + dofs_node_btwn_coordinates[i][index]]
                
                int_matrix_node = self.create_interpolation_matrix(dofs_node_btwn_coordinates[0],dofs_node_btwn_coordinates[1],interp_type)
                self.matrix_manipulation_vector(self.sorted_indices[0][1:], self.sorted_indices[1][1:], self.sorted_indices[2][1:], self.sorted_indices[3][1:], int_matrix_node[1:,1:])
            
        elif isinstance(self.functionspace.ufl_element(), fenics.MixedElement):
            counter = 0

            for subspace_index in range(self.functionspace.num_sub_spaces()):
                interp_type = self.get_interp_type(self.functionspace.sub(subspace_index))

                if self.functionspace.sub(subspace_index).num_sub_spaces() == 0:
                    if interp_type == 'linear':
                        dofs_node_btwn_coordinates = [[0],[0]]
                        
                        for i in range(2):
                            for index, dof in enumerate(self.sorted_indices[2*counter+i]):
                                if index < len(self.sorted_indices[2*counter+i])-1:
                                    dofs_node_btwn_coordinates[i] += [((self.dof_coord[dof][1] - self.dof_coord[self.sorted_indices[2*counter+i][index+1]][1])**2
                                                                    +(self.dof_coord[dof][0] - self.dof_coord[self.sorted_indices[2*counter+i][index+1]][0])**2)**(1/2) 
                                                                    + dofs_node_btwn_coordinates[i][index]]
                        
                        int_matrix_node = self.create_interpolation_matrix(dofs_node_btwn_coordinates[0],dofs_node_btwn_coordinates[1],interp_type)
                        self.matrix_manipulation_scalar(self.sorted_indices[2*counter], self.sorted_indices[2*counter+1], int_matrix_node)
                    
                    else:
                        dofs_node_btwn = [[],[],[],[]]
                        dofs_node_btwn_coordinates = [[0],[0],[0],[0]]

                        for i in range(2):
                            for index, dof in enumerate(self.sorted_indices[2*counter+i]):
                                if index%2 == 0:
                                    dofs_node_btwn[i*2] += [dof]
                                else:
                                    dofs_node_btwn[i*2+1] += [dof]
                        
                        for i in range(4):
                            for index, dof in enumerate(dofs_node_btwn[i]):
                                if index < len(dofs_node_btwn[i])-1:
                                    dofs_node_btwn_coordinates[i] += [((self.dof_coord[dof][1] - self.dof_coord[dofs_node_btwn[i][index+1]][1])**2
                                                                    +(self.dof_coord[dof][0] - self.dof_coord[dofs_node_btwn[i][index+1]][0])**2)**(1/2) 
                                                                    + dofs_node_btwn_coordinates[i][index]]

                        int_matrix_node = self.create_interpolation_matrix(dofs_node_btwn_coordinates[0],dofs_node_btwn_coordinates[2],interp_type)
                        int_matrix_btwn = self.create_interpolation_matrix(dofs_node_btwn_coordinates[1],dofs_node_btwn_coordinates[3],interp_type)
                        self.matrix_manipulation_scalar(dofs_node_btwn[0][1:], dofs_node_btwn[2][1:], int_matrix_node[1:,1:])
                        self.matrix_manipulation_scalar(dofs_node_btwn[1], dofs_node_btwn[3], int_matrix_btwn)
                    
                    counter += 1
                
                elif isinstance(self.functionspace.sub(subspace_index).ufl_element(), fenics.VectorElement):
                    if interp_type == 'quadratic' or interp_type == 'cubic':
                        dofs_node_btwn = [[],[],[],[],[],[],[],[]]
                        dofs_node_btwn_coordinates = [[0],[0],[0],[0]]

                        for i in range(4):
                            for index, dof in enumerate(self.sorted_indices[2*counter+i]):
                                if index%2 == 0:
                                    dofs_node_btwn[i*2] += [dof]
                                else:
                                    dofs_node_btwn[i*2+1] += [dof]

                        for i in range(4):
                            for index, dof in enumerate(dofs_node_btwn[i]):
                                if index < len(dofs_node_btwn[i])-1:
                                    dofs_node_btwn_coordinates[i] += [((self.dof_coord[dof][1] - self.dof_coord[dofs_node_btwn[i][index+1]][1])**2
                                                                    +(self.dof_coord[dof][0] - self.dof_coord[dofs_node_btwn[i][index+1]][0])**2)**(1/2) 
                                                                    + dofs_node_btwn_coordinates[i][index]]
                        
                        int_matrix_node = self.create_interpolation_matrix(dofs_node_btwn_coordinates[0],dofs_node_btwn_coordinates[2],interp_type)
                        int_matrix_btwn = self.create_interpolation_matrix(dofs_node_btwn_coordinates[1],dofs_node_btwn_coordinates[3],interp_type)
                        self.matrix_manipulation_vector(dofs_node_btwn[0], dofs_node_btwn[2], dofs_node_btwn[4], dofs_node_btwn[6], int_matrix_node)
                        self.matrix_manipulation_vector(dofs_node_btwn[1], dofs_node_btwn[3], dofs_node_btwn[5], dofs_node_btwn[7], int_matrix_btwn)
                    
                    elif interp_type == 'linear':
                        dofs_node_btwn_coordinates = [[0],[0]]

                        for i in range(2):
                            for index, dof in enumerate(self.sorted_indices[2*counter+i]):
                                if index < len(self.sorted_indices[2*counter+i])-1:
                                    dofs_node_btwn_coordinates[i] += [((self.dof_coord[dof][1] - self.dof_coord[self.sorted_indices[2*counter+i][index+1]][1])**2
                                                                    +(self.dof_coord[dof][0] - self.dof_coord[self.sorted_indices[2*counter+i][index+1]][0])**2)**(1/2) 
                                                                    + dofs_node_btwn_coordinates[i][index]]
                        
                        int_matrix_node = self.create_interpolation_matrix(dofs_node_btwn_coordinates[0],dofs_node_btwn_coordinates[1],interp_type)
                        self.matrix_manipulation_vector(self.sorted_indices[2*counter], self.sorted_indices[2*counter+1], self.sorted_indices[2*counter+2], self.sorted_indices[2*counter+3], int_matrix_node)
        
                    counter += 2

        # Matrix zurückgeben
        self.matrix = scipy2petsc(self.matrix_scipy, self.mesh.mpi_comm())
        A = fenics.PETScMatrix(self.matrix)
        b = fenics.PETScVector(self.vector)
        return A, b