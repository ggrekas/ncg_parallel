def line_search_wolfe1(f, fprime, xk, pk, gfk=None,
                       old_fval=None, old_old_fval=None,
                       args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
                       xtol=1e-14):


    """
As `scalar_search_wolfe1` but do a line search to direction `pk`
Parameters
----------
f : callable
    Function `f(x)`
fprime : callable
    Gradient of `f`
xk : array_like
    Current point
pk : array_like
    Search direction
gfk : array_like, optional
    Gradient of `f` at point `xk`
old_fval : float, optional
    Value of `f` at point `xk`
old_old_fval : float, optional
    Value of `f` at point preceding `xk`
The rest of the parameters are the same as for `scalar_search_wolfe1`.
Returns
-------
stp, f_count, g_count, fval, old_fval
    As in `line_search_wolfe1`
gval : array
    Gradient of `f` at the final point
"""
    if gfk is None:
        gfk = fprime(xk) # return vector

    if isinstance(fprime, tuple):
        eps = fprime[1]
        fprime = fprime[0]
        newargs = (f, eps) + args
        gradient = False
    else:
        newargs = args
        gradient = True

    gval = [gfk]
    gc = [0]
    fc = [0]


    def phi(s):
        fc[0] += 1
        xk_c = xk.copy()
        xk_c.axpy(s, pk)
        return f(xk_c, *args) # modify me


    def derphi(s):
        xk_c = xk.copy()
        xk_c.axpy(s, pk)
        gval[0] = fprime(xk_c, *newargs) # modify me
        if gradient:
            gc[0] += 1
        else:
            fc[0] += len(xk) + 1 # modify me
        return gval[0].inner(pk)#np.dot(gval[0], pk) # modify me


    derphi0 = gfk.inner(pk) #np.dot(gfk, pk) # modify me

    stp, fval, old_fval = scalar_search_wolfe1(
    phi, derphi, old_fval, old_old_fval, derphi0,
    c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

    return stp, fc[0], gc[0], fval, old_fval, gval[0]


def scalar_search_wolfe1(phi, derphi, phi0=None, old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9,
                         amax=50, amin=1e-8, xtol=1e-14):
    """
    Scalar function search for alpha that satisfies strong Wolfe conditions
    alpha > 0 is assumed to be a descent direction.
    Parameters
    ----------
    phi : callable phi(alpha)
        Function at point `alpha`
    derphi : callable dphi(alpha)
        Derivative `d phi(alpha)/ds`. Returns a scalar.
    phi0 : float, optional
        Value of `f` at 0
    old_phi0 : float, optional
        Value of `f` at the previous point
    derphi0 : float, optional
        Value `derphi` at 0
    c1, c2 : float, optional
        Wolfe parameters
    amax, amin : float, optional
        Maximum and minimum step size
    xtol : float, optional
        Relative tolerance for an acceptable step.
    Returns
    -------
    alpha : float
        Step size, or None if no suitable step was found
    phi : float
        Value of `phi` at the new point `alpha`
    phi0 : float
        Value of `phi` at `alpha=0`
    Notes
    -----
    Uses routine DCSRCH from MINPACK.
    """

    if phi0 is None:
        phi0 = phi(0.)
    if derphi0 is None:
        derphi0 = derphi(0.)

    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01 * 2 * (phi0 - old_phi0) / derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0

    phi1 = phi0
    derphi1 = derphi0
    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b'START'

    maxiter = 100
    for i in xrange(maxiter):
        stp, phi1, derphi1, task = minpack2.dcsrch(alpha1, phi1, derphi1,
                                                   c1, c2, xtol, task,
                                                   amin, amax, isave, dsave)
        if task[:2] == b'FG':
            alpha1 = stp
            phi1 = phi(stp)
            derphi1 = derphi(stp)
        else:
            break
    else:
        # maxiter reached, the line search did not converge
        stp = None

    if task[:5] == b'ERROR' or task[:4] == b'WARN':
        stp = None  # failed

    return stp, phi1, phi0


line_search = line_search_wolfe1
