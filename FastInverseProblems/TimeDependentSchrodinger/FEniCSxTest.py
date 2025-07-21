from mpi4py import MPI
import dolfinx
import dolfinx.plot as plot
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import ufl

import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_MESA"] = "true"
import pyvista as pv
#pv.OFF_SCREEN = True

# For complex-number support
from petsc4py import PETSc


# --- Setup mesh, spaces & functions ---------------------------------------

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

u_r = dolfinx.fem.Function(V, dtype=np.float64)
u_r.interpolate(lambda x: x[0])
u_c = dolfinx.fem.Function(V, dtype=np.complex128)
u_c.interpolate(lambda x: 0.5 * x[0]**2 + 1j * x[1]**2)

print(u_r.x.array.dtype)
print(u_c.x.array.dtype)

print("PETSc ScalarType:", PETSc.ScalarType)
assert np.dtype(PETSc.ScalarType).kind == "c"

# --- Variational problem & solve -----------------------------------------

x = ufl.SpatialCoordinate(mesh)
f = dolfinx.fem.Constant(mesh, PETSc.ScalarType(-1 - 2j))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

topo_facets = mesh.topology
topo_facets.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(
    V, mesh.topology.dim - 1, boundary_facets)
bc = dolfinx.fem.dirichletbc(u_c, boundary_dofs)

problem = LinearProblem(a, L, bcs=[bc])
uh = problem.solve()

u_ex = 0.5 * x[0]**2 + 1j * x[1]**2
error_form = dolfinx.fem.form(
    ufl.dot(uh - u_ex, uh - u_ex) * ufl.dx(metadata={"quadrature_degree": 5})
)
local_error = dolfinx.fem.assemble_scalar(error_form)
global_error = np.sqrt(mesh.comm.allreduce(local_error, op=MPI.SUM))
max_error = mesh.comm.allreduce(np.max(np.abs(u_c.x.array - uh.x.array)))
print("L2 error:", global_error)
print("Max nodal error:", max_error)

# --- PyVista Visualization -----------------------------------------------

# Extract mesh for PyVista
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
topo, cell_types, geometry = plot.vtk_mesh(mesh, mesh.topology.dim)

grid = pv.UnstructuredGrid(topo, cell_types, geometry)
grid.point_data["u_real"] = uh.x.array.real
grid.point_data["u_imag"] = uh.x.array.imag
grid.set_active_scalars("u_real")

# Off-screen rendering
#pv.OFF_SCREEN = True
plotter = pv.Plotter(off_screen=True)
plotter.add_text("uh real", position="upper_edge", font_size=14, color="black")
plotter.add_mesh(grid, show_edges=True, cmap="viridis")
plotter.view_xy()
img_array = plotter.screenshot("realSoln.png")
plotter.close()

print("Saved realSoln.png; img_array shape:", img_array.shape)
