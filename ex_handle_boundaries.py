from dolfin import *
from init_u import*
import numpy as np
import matplotlib.pyplot as plt
import time as mtime

# Optimization options for the form compiler
#parameters["form_compiler"]["precision"] = 100
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
#mesh = UnitCubeMesh(16, 12, 12)

k=1
epsilon = None
nx, ny = 100, 100
mesh = UnitSquareMesh(nx, ny)



V = VectorFunctionSpace(mesh, "Lagrange", 1)
W = FunctionSpace(mesh, "Lagrange", 1)

# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)


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
psi = (mu/2)*(Ic- 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2

print k, epsilon, 'n = ', nx, ny
u0 = Function(V)
for bc in bcs:
    bc.apply(u0.vector() )


Pi =  psi*dx 


DP2 = derivative(Pi, u, v)
start_time = mtime.time()


#u_array = initialization(psi, V, bcs, du, v, u)
i = 0


def free_indices_num(free_i):
    """
    Creates and returns a vector which in the index mpiRank contains
    the number len(free_i)
    """
    comm = mpi_comm_world()
    mpiRank = MPI.rank(comm)
    import numpy as np

    free_i_num_local = PETScVector()
    free_i_num_local.init(comm, (mpiRank, mpiRank+1) )
    free_i_num_local.set_local(np.array([len(free_i)], dtype='double') )

    free_i_num = PETScVector()
    mpiSize = MPI.size(comm)
    free_i_num_local.gather(free_i_num,
                            np.linspace(0, mpiSize-1, mpiSize).astype('intc') )

    return free_i_num

def create_reduced_vector(free_i_num):
    comm = mpi_comm_world()
    mpiRank = MPI.rank(comm)
    mpiSize = MPI.size(comm)
    # import numpy as np

    x_new = PETScVector()
    free_i_num_ar = free_i_num.array()
    f_i = free_i_num_ar.astype('intc')


    prev_sum = sum(f_i[:mpiRank])
    print 'mpiRank =', mpiRank, ', sum =', prev_sum, ', free_i=', free_i_num_ar[mpiRank]
    x_new.init(comm, (prev_sum, int(prev_sum + f_i[mpiRank]) ) )

    return x_new


bc_temp1 = DirichletBC(V, Expression(('1.0', '1.0')), left)
bc_temp2 = DirichletBC(V, Expression(('1.0', '1.0')), right)
u_temp = Function(V)
bcs_temp = [bc_temp1, bc_temp2]
for bc in bcs_temp:
    bc.apply(u.vector() )

free_ind = np.nonzero(u.vector().array()==0.0 )
free_i = free_ind[0]; free_i = free_i.astype('intc')
n_free_ind = np.nonzero(u.vector().array()!=0.0 )
n_free_i = n_free_ind[0]



free_i_num = free_indices_num(free_i)
x0 = create_reduced_vector(free_i_num)
x_temp = create_reduced_vector(free_i_num)


for bc in bcs:
    bc.apply(u.vector() )



aPiMax =assemble(Pi)
aPiMax = 100
print 'aPi = ', assemble(Pi)	

def f3(x, *args):
    # u.vector()[free_i] = x
    u.vector().set_local(x.array(), free_i)
    u.vector().apply('')
    aPi = assemble(Pi)

    if np.isnan(aPi):
        print aPi
        return aPiMax
    indx_ar = args[0]
    en_ar = args[1]
    en_ar[indx_ar[0]] = aPi

    indx_ar[0] +=1
    print indx_ar[0], aPi

    return aPi

def grad_f3(x, *args):
    mpiRank = MPI.rank(comm)
    # u.vector()[free_i] = x
    u.vector().set_local(x.array(), free_i)
    u.vector().apply('')
    x_temp.set_local(assemble(DP2).array()[free_i])

    x_temp.apply('')
    return x_temp.copy()



u_array2 = u0.vector().array()

class myf(Expression):
    def eval(self,value, x):
        value[0] = 0.2*x[0]
        value[1] = 0.0
    def value_shape(self):
        return (2,)
bExpr = Expression(('0.2*x[0]', '0'))
f_p = project(myf(), V)
#plot(f_p); interactive()
u.assign(f_p)
u_array = u.vector().array()
x0.set_local(u_array[free_i])
x0.apply('')
N = u_array.shape[0]
en = np.zeros(50000)

i_ar = np.array([0])
#args = [j]
args = (i_ar, en)

warnflag = 2
i=0

comm = mpi_comm_world()
mpiRank =MPI.rank(comm)
v1 = u_temp.vector()
if mpiRank == 0:
    #print 'in 0', v1.array()
    print '0: global=', v1.size() ,'local size', v1.local_size()
else:
   # print 'in 1', v1.array()
    print '1: global=', v1.size() ,'local size', v1.local_size()
    


from ncg_PETS import*
x0.apply('')
u_v = minimize_cg(f3, x0, grad_f3, args=args)

u.vector().set_local(u_v.array(), free_i)
# import time
print 'time to create-solve problem = ', mtime.time()-start_time, 'seconds'
# plot(u, interactive=True, mode='displacement')

'''
u_array, fopt, f_calls, g_calls, warnflag = optimize.fmin_cg(f3, u_array[free_i], fprime = grad_f3, args = args, full_output=1)
#u_array2, fopt, f_calls, g_calls, warnflag = optimize.fmin_cg(f, u_array2, fprime = grad_f2, args = args, full_output=1)


while warnflag == 2:
    u_array, fopt, f_calls, g_calls, warnflag = optimize.fmin_cg(f3, u_array, fprime = grad_f3, args = args, full_output=1)
    i+=1
    if i==100:
        break
u.vector()[free_i] = u_array
#u_array = optimize.anneal(f, x.astype('float128') )

#for bc in bcs:
#	bc.apply(u.vector() )



#plt.plot(en[:i_ar[0] ])
#plt.show()


#for bc in bcs:
#	bc.apply(u.vector() )




# Plot and hold solution
print 'Energy = ', assemble(Pi), 'k = ', k, 'n = ', nx, ny

s = 'k=' + str(k) + '_nx=' + str(nx) + '_ny=' + str(ny)
plot(u, mode = "displacement", interactive = True, title = s)
'''
