
# Begin demo

from dolfin import *
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from ncg_PETS import*

import time

# Optimization options for the form compiler
parameters["mesh_partitioner"] = "SCOTCH"  # "ParMETIS" #

# parameters["form_compiler"]["precision"] = 1000
parameters["allow_extrapolation"]=True
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = \
"-O3 -ffast-math -march=native"
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
#
# Create mesh and define function space
#mesh = UnitCubeMesh(16, 12, 12)

k=10
epsilon = None
nx, ny = 100, 100
mesh = UnitSquareMesh(nx, ny)
import mshr
domain = mshr.Rectangle(dolfin.Point(0, 0), dolfin.Point(1, 1))
mesh = mshr.generate_mesh(domain, 64)


boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)


class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0]) < tol

Gamma_0 = LeftBoundary()
Gamma_0.mark(boundary_parts, 0)


class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0] - 1) < tol

Gamma_1 = RightBoundary()
Gamma_1.mark(boundary_parts, 1)


class UpDown(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and (abs(x[1]) < tol or abs(1 - x[1]) < tol) 

Gamma_2 = UpDown()
Gamma_2.mark(boundary_parts, 2)



V = VectorFunctionSpace(mesh, "Lagrange", 1)
W = FunctionSpace(mesh, "Lagrange", 1)

# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

'''
# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(("0.0", "0.0", "0.0"))
r = Expression(("scale*0.0",
                "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
                "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
                scale = 0.5, y0 = 0.5, z0 = 0.5, theta = pi/3)
'''

c = Expression(("0.0", "0.0"))
r = Expression(("0.2", "0.0"))


bcl = DirichletBC(V, c, left)
bcr = DirichletBC(V, r, right)
bcs = [bcl, bcr]
#bcs = [DirichletBC(V, c, boundary_parts, 0),
#	       DirichletBC(V, r, boundary_parts, 1)]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
#T  = Constant((0.1,  0.0, 0.0))  # Traction force on the boundary

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = variable(I + grad(u) )             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2 


print k, epsilon, 'n = ', nx, ny
u0 = Function(V)
for bc in bcs:
    bc.apply(u0.vector() )



k_f = Constant('1.0')
k_p = Constant(str(1.0/k) )
#k_f = Constant('50.0')
#k_p = Constant('1.0')

Pi =  k_p*psi*dx + k_f*dot(u, u)*ds(0, subdomain_data=boundary_parts) + \
    k_f*dot(u-u0, u-u0)*ds(1, subdomain_data=boundary_parts)


DP = derivative(Pi, u, v)

start_time = time.time()


#u_array = initialization(psi, V, bcs, du, v, u)
i = 0

#gf = np.zeros( u.vector().array().shape[0] )
comm = mpi_comm_world()
dofmap = V.dofmap()
my_first, my_last = dofmap.ownership_range()                # global

# 'Handle' API change of tabulate coordinates
if dolfin_version().split('.')[1] == '7':
    x = V.tabulate_dof_coordinates().reshape((-1, 2))
else:
    x = V.dofmap().tabulate_all_coordinates(mesh)

unowned = dofmap.local_to_global_unowned()
dofs = filter(lambda dof: dofmap.local_to_global_index(dof) not in unowned,
              xrange(my_last-my_first))



def f(x, *args):
    # u.vector().set_local(x.array())
    u.vector()[:] = x.array()
    # u.vector().apply('insert')
    u.vector().apply('')

    aPi = assemble(Pi)

    if np.isnan(aPi):
        print aPi
        return aPi
    indx_ar = args[0]
    en_ar = args[1]
    en_ar[indx_ar[0]] = aPi

    indx_ar[0] +=1

    mpiRank = MPI.rank(comm)
    if 0 == mpiRank:
        print indx_ar[0], aPi

    return aPi

def grad_f(x, *args):
    # print 'in grad f', x
    u.vector()[:] = x.array()
    # u.vector().set_local(x.array())
    # u.vector().apply('insert')
    u.vector().apply('')
    DP_v = assemble(DP)

    # print 'grad f returns'
    return DP_v

def f2(x, *args):

    u.vector().set_local(np.array(x))
    aPi = assemble(Pi)

    if np.isnan(aPi):
        print aPi
        return aPi
    indx_ar = args[0]
    en_ar = args[1]
    en_ar[indx_ar[0]] = aPi

    indx_ar[0] +=1

    mpiRank = MPI.rank(comm)
    if 0 == mpiRank:
        print indx_ar[0], aPi

    return aPi

def grad_f2(x, *args):
    u.vector().set_local( x )
    DP_v = assemble(DP)

    return DP_v.array()



x = u.vector()
# as_backend_type(u.vector()).update_ghost_values()
mpiSize =MPI.size(comm)
N = u.vector().size()
en = np.zeros(50000)

i_ar = np.array([0])
#args = [j]
args = (i_ar, en)

# x = as_backend_type(x)
# u_v = optimize.fmin_cg(f, x, fprime = grad_f, args = args)
x0 = PETScVector(comm, N)
# x.set_local(x0.array())
# x =as_backend_type(x)

size = (u.vector().size())
l_size = (u.vector().local_size())
x.apply('')
x0 = x.copy()
print 'u size=', size, 'l u size=', l_size

size = (x0.size())
l_size = (x0.local_size())

print 'x0 size=', size, 'l x0 size=', l_size


u_v = minimize_cg(f, x0, grad_f, args=args)
# x.apply('')
# u_v = minimize_cg(f2, x, grad_f2, args=args)
#u_array = optimize.anneal(f, x.astype('float128') )

#for bc in bcs:
#	bc.apply(u.vector() )

# u.vector().set_local(np.array(u_v.array()))
# plt.plot(en[:i_ar[0] ])
# plt.show()
#
#
# #for bc in bcs:
# #	bc.apply(u.vector() )


print 'time to create-solve problem = ', time.time()-start_time, 'seconds'

# Plot and hold solution
print 'Energy = ', assemble(Pi), 'k = ', k, 'n = ', nx, ny

# s = 'k=' + str(k) + '_nx=' + str(nx) + '_ny=' + str(ny)
# plot(u, mode = "displacement", interactive = True, title = s)
