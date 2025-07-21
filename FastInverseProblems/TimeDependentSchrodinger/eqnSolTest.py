import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import assemble_matrix, create_matrix
from petsc4py import PETSc
from ufl import dx, grad, inner, conj
import basix.ufl

import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_MESA"] = "true"
import pyvista as pv
#pv.OFF_SCREEN = True

# Create 1D mesh
initialX = 0.0
finalX = 1.0
domain = mesh.create_interval(MPI.COMM_WORLD, 500, [initialX, finalX])
# Choose correct cell type string
cell = domain.topology.cell_type.name  # "interval"
#V = fem.FunctionSpace(domain, ("CG", 1), dtype=complex)
element = basix.ufl.element("CG", cell, 1)
V = fem.functionspace(domain, element)

# Time parameters
dt = 1e-3
T = 0.1
num_steps = int(T / dt)

# Initial condition: Gaussian
x = ufl.SpatialCoordinate(domain)



#IC = ufl.exp(-100.0 * (x[0])**2)
IC = ufl.exp(-100.0 * (x[0]-0.5)**2)

psi_0_expr = fem.Expression(IC,V.element.interpolation_points(),comm=MPI.COMM_WORLD)
psi_n = fem.Function(V)
psi_n.interpolate(psi_0_expr)

# Trial and test functions
psi = ufl.TrialFunction(V)
u = ufl.TestFunction(V)

# Potential V(x) = 0 (free particle)
V_pot = fem.Function(V)
V_pot.x.array[:] = 0.0  # Could also use a nonzero potential

# Crank-Nicolson average
psi_mid = 0.5 * (psi + psi_n)

# Weak form
a = (1j * inner(psi, u) / dt + 0.5 * inner(grad(psi), grad(u)) + inner(V_pot * psi, u)) * dx
L = (1j * inner(psi_n, u) / dt - 0.5 * inner(grad(psi_n), grad(u)) - inner(V_pot * psi_n, u)) * dx


# Apply Dirichlet BCs (ψ = 0 on boundary)
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0))
bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(0 + 0j, bc_dofs, V)

# Assemble system
#A = fem.petsc.assemble_matrix(fem.form(a))
#A.assemble()
a_form = fem.form(a)  # your UFL bilinear form
A = create_matrix(a_form)            # allocate PETSc matrix
assemble_matrix(A, a_form, bcs=[bc])  # fill it
A.assemble()                
b = fem.petsc.create_vector(fem.form(L))

fem.petsc.apply_lifting(b, [fem.form(a)], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Prepare solution function
psi_sol = fem.Function(V)

'''
# Time-stepping loop
for n in range(num_steps):
    b = fem.petsc.create_vector(fem.form(L))
    fem.petsc.assemble_vector(b, fem.form(L))
    fem.petsc.apply_lifting(b, [fem.form(a)], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc])
    fem.petsc.solve(A, psi_sol.vector, b)
    psi_sol.x.scatter_forward()
    psi_n.x.array[:] = psi_sol.x.array[:]
'''

coords = domain.geometry.x[:, 0]
nx = coords.size
nsteps = num_steps + 1  # if including t=0

psi_data = np.zeros((nsteps, nx), dtype=complex)
psi_data[0, :] = psi_n.x.array[:nx]  # initial state

# Initialize the solver outside the loop
problem = LinearProblem(a, L, bcs=[bc], u=psi_sol,petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# Time-stepping
for n in range(num_steps):
    psi_sol = problem.solve()
    psi_n.x.array[:] = psi_sol.x.array[:]
    psi_data[n+1, :] = psi_sol.x.array[:nx]
    
# Postprocessing (export)
if MPI.COMM_WORLD.rank == 0:
    import matplotlib.pyplot as plt
    '''
    dof_coords = V.tabulate_dof_coordinates().reshape((-1,))
    plt.plot(dof_coords, np.abs(psi_sol.x.array)**2)
    plt.xlabel("x")
    plt.ylabel(r"$|\psi(x,T)|^2$")
    plt.title("Final Probability Density")
    plt.grid(True)
    plt.savefig("Final Probability Density.png")
    '''
    coords = domain.geometry.x[:, 0]  # shape (101,)
    vals = np.abs(psi_sol.x.array[:coords.size])**2

    plt.plot(coords, vals, '.-')
    plt.xlabel("x")
    plt.ylabel(r"$|\psi(x,T)|^2$")
    plt.title("Final Probability Density")
    plt.grid(True)
    plt.savefig("Final Probability Density.png")

    tvals = np.arange(nsteps) * dt
    xvals = coords

    real = psi_data.real
    imag = psi_data.imag
    mag  = np.abs(psi_data)

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    def make_image(ax, data, title, cmap):
        im = ax.imshow(data,
                    origin='lower',
                    aspect='auto',
                    extent=[xvals[0], xvals[-1], tvals[0], tvals[-1]],
                    cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)

    make_image(axes[0], real, "Re(ψ(t,x))", 'RdBu_r')
    make_image(axes[1], imag, "Im(ψ(t,x))", 'RdBu_r')
    make_image(axes[2], mag,  "|ψ(t,x)|", 'viridis')

    plt.tight_layout()
    plt.savefig("Schrodinger_solution.png", dpi=300) 