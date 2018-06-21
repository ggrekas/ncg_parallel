from dolfin import*
import warnings
# from scipy.optimize import linesearch
import myPar_line_search as linesearch


def minimize_cg(fun, x0, fprime=None, args=(), callback=None,
                 gtol=1e-5, norm='linf', maxiter=None,
                 disp=True, return_all=False,
                 **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    conjugate gradient algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    """
    f = fun
    warnflag = 0

    if maxiter is None:
        maxiter = x0.size() * 200

    func_calls, f = wrap_function(f, args)
    grad_calls, myfprime = wrap_function(fprime, args)


    comm = mpi_comm_world()
    # gfk_ar = myfprime(x0.array()) #change meeeeeee
    # gfk = PETScVector(comm, x0.size())
    # gfk.set_local(gfk_ar)
    gfk = myfprime(x0).copy()  # change meeeeeee

    k = 0
    xk = x0

    # Sets the initial step guess to dx ~ 1
    old_fval = f(xk)
    old_old_fval = old_fval + gfk.norm('l2') / 2.0 #None #

    pk = x0.copy()
    pk.set_local(-gfk.get_local() ) #pk = -gfk

    gnorm = gfk.norm('linf')  #gnorm = vecnorm(gfk, ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        deltak = gfk.inner(gfk) #numpy.dot(gfk, gfk)

        try:
            # xk0 = xk.gather_on_zero()
            # pk0 = pk.gather_on_zero()
            # gfk0 = gfk.gather_on_zero()

            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                    old_fval, old_old_fval, c2=0.4)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        xk.axpy(alpha_k,pk)     #xk = xk + alpha_k * pk

        if gfkp1 is None:
            gfkp1 = myfprime(xk)  #malloc?

        # is the substruction in set local faster or slower than axpy
        gfk.set_local(gfkp1.get_local() - gfk.get_local())
        beta_k = max(0, gfk.inner(gfkp1) / deltak)

        pk.set_local(-gfkp1.get_local() + beta_k * pk.get_local() )
        #pk = -gfkp1 + beta_k * pk
        gfk.set_local(gfkp1.get_local()) #del gfk; gfk = gfkp1

        gnorm = gfk.norm(norm) #gnorm = vecnorm(gfk, ord=norm)
        if callback is not None:
            callback(xk)
        k += 1

    fval = old_fval

    mpiRank = MPI.rank(comm)
    if 0 == mpiRank:
        _print_result_info(warnflag, fval, k, maxiter, disp)

    del pk
    if return_all == True:
        return xk, warnflag
    return xk


def _print_result_info(warnflag, fval, k, maxiter, disp=False):
    func_calls = [0]; grad_calls = [0]

    if warnflag == 2:
        msg = _status_message['pr_loss']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])

    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])


    return

            # standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                  'maxfev': 'Maximum number of function evaluations has '
                  'been exceeded.',
                  'maxiter': 'Maximum number of iterations has been '
                  'exceeded.',
                  'pr_loss': 'Desired error not necessarily achieved due '
                  'to precision loss.'}


def line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval, **kwargs):

    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.
    Raises
    ------
    _LineSearchError
        If no suitable step size is found
    """
    ret = linesearch.line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    if ret[0] is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', linesearch.LineSearchWarning)
            ret = linesearch.line_search_wolfe2(f, fprime, xk, pk, gfk,
                                                old_fval, old_old_fval)

    if ret[0] is None:
        raise _LineSearchError()

    return ret



class _LineSearchError(RuntimeError):
    pass

def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper



