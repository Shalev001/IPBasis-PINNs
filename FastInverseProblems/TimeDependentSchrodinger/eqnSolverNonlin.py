import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, plot
import dolfinx_mpc
#from dolfinx_mpc import LinearProblem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import assemble_matrix, create_matrix
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from ufl import dx, grad, inner, conj
import basix.ufl

import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_MESA"] = "true"
import pyvista as pv
#pv.OFF_SCREEN = True

# Create 1D mesh
initialX = -5
finalX = 5
domain = mesh.create_interval(MPI.COMM_WORLD, 5000, [initialX, finalX])
# Choose correct cell type string
cell = domain.topology.cell_type.name  # "interval"
#V = fem.FunctionSpace(domain, ("CG", 1), dtype=complex)
#element = basix.ufl.element("CG", cell, 1)
element = basix.ufl.element("Lagrange",cell,1)
V = fem.functionspace(domain, element)

'''
mpc = dolfinx_mpc.MultiPointConstraint(V)

left = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], initialX))
right = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], finalX))
mpc.create_periodic_constraint_geometrical(V, lambda x: np.isclose(x[0], finalX),lambda x: x - [finalX - initialX], [])
mpc.finalize()
'''

# Time parameters
dt = 5e-3
T = 3.0
num_steps = int(T / dt)

# Initial condition: Gaussian
x = ufl.SpatialCoordinate(domain)

#IC = ufl.exp(-100.0 * (x[0])**2)
#IC = 2*ufl.cosh((x[0]-offset))**-1

kx    = 0.1                        # wave number
m     = 1                          # mass
sigma = 0.5                   # width of initial gaussian wave-packet
x0    = 0.0                     # center of initial gaussian wave-packet

A = 1.0 / (sigma * np.sqrt(np.pi)) # normalization constant

# Initial Wavefunction
IC = ufl.sqrt(A) * ufl.exp(-(x[0]-x0)**2 / (2.0 * sigma**2)) * ufl.exp(1j * kx * x[0])

#np.exp(-(x-x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kx * x)

psi_0_expr = fem.Expression(IC,V.element.interpolation_points(),comm=MPI.COMM_WORLD)
psi_n = fem.Function(V)
psi_n.interpolate(psi_0_expr)

psi = fem.Function(V)         # nonlinear unknown
u = ufl.TestFunction(V)

# Potential V(x) = 0 (free particle)
'''k=1
V_pot = fem.Function(V)

if k != 0:
    #if k=0 this expression breaks
    V_expr = fem.Expression(0.5 * k * x[0]**2 +0j, V_pot.function_space.element.interpolation_points())
    V_pot.interpolate(V_expr)

if k==0:
    V_pot.x.array[:] = 0.0'''

# 2. Build forms
V_pot_n = ufl.conj(psi_n) * psi_n
a = (
    1j * inner(psi, u) / dt
    + 0.5 * inner(ufl.grad(psi), ufl.grad(u))
    + inner((V_pot_n * psi), u)
) * dx


L = (
    1j * inner(psi_n, u) / dt
    - 0.5 * inner(ufl.grad(psi_n), ufl.grad(u))
    - inner((V_pot_n * psi_n), u)
) * dx

#V_pot.x.array[:] = np.abs(psi.x.array[:])**2


# Crank-Nicolson average
#psi_mid = 0.5 * (psi + psi_n)

# Weak form
'''a = (1j * inner(psi, u) / dt + 0.0003636947 * inner(grad(psi), grad(u)) + inner(V_pot * psi, u)) * dx
L = (1j * inner(psi_n, u) / dt - 0.0003636947 * inner(grad(psi_n), grad(u)) - inner(V_pot * psi_n, u)) * dx'''

'''a = (1j * inner(psi, u) / dt + 0.5*inner(grad(psi), grad(u)) + inner(V_pot * psi, u)) * dx
L = (1j * inner(psi_n, u) / dt - 0.5*inner(grad(psi_n), grad(u)) - inner(V_pot * psi_n, u)) * dx'''


# Apply Dirichlet BCs (ψ = 0 on boundary)
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], initialX) | np.isclose(x[0], finalX))
bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(0 + 0j, bc_dofs, V)

'''
# Assemble system
#A = fem.petsc.assemble_matrix(fem.form(a))
#A.assemble()
a_form = fem.form(a)  # your UFL bilinear form
A = create_matrix(a_form)            # allocate PETSc matrix
assemble_matrix(A, a_form, bcs=[bc])  # fill it
A.assemble()                
b = fem.petsc.create_vector(fem.form(L))

fem.petsc.apply_lifting(b, [fem.form(a)], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)'''

# Prepare solution function
#psi_sol = fem.Function(mpc.function_space)
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
nsteps = num_steps + 1  # if including t=0

# This gives you the true number of DOFs (complex-valued):
num_dofs = V.dofmap.index_map.size_local #// 2

psi_data = np.zeros((nsteps, num_dofs), dtype=complex)
psi_data[0, :] = psi_n.x.array[:num_dofs]  # initial state

# Initialize the solver outside the loop
#problem = LinearProblem(a, L,mpc,bcs=None, u=psi_sol,petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
#problem = LinearProblem(a, L,bcs=[bc], u=psi_sol,petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# 3. Residual and Jacobian
F = a - L
J = ufl.derivative(F, psi)
problem = NonlinearProblem(F, psi, bcs=[bc], J=J)
solver = NewtonSolver(domain.comm, problem)
solver.set_form(lambda x: problem.form(x))

# Time-stepping
for n in range(num_steps):
    t = (n+1) * dt

    psi.x.array[:] = psi_n.x.array.copy()
    
    # Recompute pi-dependent potential and forms
    V_pot_n = ufl.conj(psi_n) * psi_n

    a = (
        1j * inner(psi, conj(u)) / dt
        + 0.5 * inner(grad(psi), grad(conj(u)))
        + inner(V_pot_n * psi, conj(u))
    ) * dx

    L = (
        1j * inner(psi_n, conj(u)) / dt
        - 0.5 * inner(grad(psi_n), grad(conj(u)))
        - inner(V_pot_n * psi_n, conj(u))
    ) * dx

    F = a - L
    J = ufl.derivative(F, psi)

    # Update problem with new forms and Jacobian if necessary
    problem.F = F
    problem.J = J

    # Solve nonlinear step
    niter, converged = solver.solve(psi)

    if not converged:
        raise RuntimeError(f"Newton failed at step {n+1}, t={t}")

    # Update for next step
    psi_n.x.array[:] = psi.x.array

    psi_data[n+1, :] = psi.x.array[:num_dofs]

#saving collected data
#np.save("data1.npy",psi_data)
    
# Postprocessing (export)
if MPI.COMM_WORLD.rank == 0:
    print("plotting")
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

    # Get tabulated coordinates
    coords_raw = V.tabulate_dof_coordinates().real  # shape (15002, 1)

    # Reshape and extract only one entry per DOF
    # Safe version: take first num_dofs entries (they match .x.array[:])
    dof_coords = coords_raw[:num_dofs, 0]  # shape: (10001,)

    for i in range(0,psi_data.shape[0],1):
        vals = np.abs(psi_data[i,:])**2
        plt.plot(np.sort(dof_coords), vals[np.argsort(dof_coords)], '.-')
    plt.xlabel("x")
    plt.ylabel(r"$|\psi(x,T)|^2$")
    plt.title("Probability Density")
    plt.grid(True)
    plt.savefig("Probability Density.png")

    tvals = np.arange(nsteps) * dt
    xvals = V.tabulate_dof_coordinates().real[:2 * num_dofs:2, 0]
    xvals = np.sort(xvals)

    real = psi_data.real
    imag = psi_data.imag
    mag  = np.abs(psi_data)**2

    fig, axes = plt.subplots(3, 1, figsize=(4, 12), sharex=True)

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

    