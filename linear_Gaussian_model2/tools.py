"""Tools for linalg comps"""
import scipy.linalg as sc
import numpy as np
#from jax.numpy import matmul as gmatmul
#import numba


def mat_mul(mat1, mat2):
    """Matrix multiplication"""
    return np.matmul(mat1, mat2)
    #result = gmatmul(mat1, mat2)
    #return np.array(result)

def mvnrnd(mean, cov, coln=1):
    """ multivariate_normal (faster than the numpy built-in function:
        np.random.multivariate_normal) """
    mndim = mean.ndim
    if (mndim == 1) & (coln == 1):
        result = mean + mat_mul(sc.cholesky(cov),
                                np.random.standard_normal(mean.size))
    elif (mndim == 1) & (coln > 1):
        rand_ = np.random.standard_normal(size=(mean.size, coln))
        result = mean.reshape(-1, 1) + mat_mul(sc.cholesky(cov), rand_)
    elif mndim > 1:
        result = mean + mat_mul(sc.cholesky(cov),
                                np.random.standard_normal(size=mean.shape))
    return result

def fwd_slash(mat1, mat2):
    """ equivalent to mat1/mat2 in MATLAB. That is to solve for x in: x * mat2 = mat1
        This is equivalent to: mat2.T * x.T = mat1.T """
    return np.linalg.solve(mat2.T, mat1.T).T

def bisection(func, params, point_a=0., point_b=1.):
    """Find zeros by bisection"""
    bisection_nmax = params["bisection_nmax"]
    error = params["bisection_error"]
    diff = params["bisection_diff"]
    func_a = func(point_a)
    func_b = func(point_b)
    if func_a * func_b >= 0:
        #msg = "f(a) and f(b) must have different signs\n"
        #msg = msg + " f(a) = %.4f, f(b) = %.4f\n" % (func_a, func_b)
        #msg = msg + 'Try different values of b\n'
        #print(msg)
        iters = 0
        point_b = point_b/20
        func_b = func(point_b)
        while (func_a * func_b >= 0) & (iters < 100):
            iters += 1
            point_b = point_b - 0.0005
            func_b = func(point_b)
        if func_a * func_b >= 0:
            zero = point_a
            converged = False
            return zero, converged
    iters = 0
    converged = False
    while ((point_b - point_a) >= diff) & (iters <= bisection_nmax):
        iters += 1
        # Find middle point
        zero = (point_a + point_b) / 2
        # Check if middle point is root
        if np.abs(func(zero)) <= error:
            converged = True
            break
        # Decide the side to repeat the steps
        if func(zero) * func(point_a) < 0.:
            point_b = zero
        else:
            point_a = zero
    if iters > bisection_nmax:
        zero = point_a
        converged = False
    return zero, converged

def logdet(matrix, chol=True):
    # pylint: disable-msg=W0632

    """ LOGDET Computation of logarithm of determinant of a matrix

      v = logdet(matrix, chol = False)
          computes the logarithm of determinant of matrix.

      v = logdet(matrix)
          If matrix is positive definite
      """
    if chol:
        vector = 2 * np.sum(np.log(np.diag(sc.cholesky(matrix))))
    else:
        prec, _, mat_u = sc.lu(matrix)
        diag_u = np.diag(mat_u)
        prec_diag_u = np.linalg.det(prec) * np.prod(np.sign(diag_u))
        vector = np.log(prec_diag_u) + np.sum(np.log(np.abs(diag_u)))
    return vector

def nearestpd(mat):
    """Find the nearest positive-definite matrix to input
    """
    mat_b = (mat + mat.T) / 2
    _, mat_s, mat_v = np.linalg.svd(mat_b)

    mat_h = np.dot(mat_v.T, np.dot(np.diag(mat_s), mat_v))

    mat2 = (mat_b + mat_h) / 2

    mat3 = (mat2 + mat2.T) / 2

    if is_positive_def(mat3):
        return mat3

    spacing = np.spacing(np.linalg.norm(mat))
    identity = np.eye(mat.shape[0])
    k = 1
    while not is_positive_def(mat3):
        mineig = np.min(np.real(np.linalg.eigvals(mat3)))
        mat3 += identity * (-mineig * k**2 + spacing)
        k += 1

    return mat3

def is_positive_def(mat):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False

def symmetric(matrix):
    """Symmetric matrix"""
    return np.triu(matrix) + np.triu(matrix, 1).T

def logmvnpdf(d,X_minus_mu,sigd): 
    """evaluate the Multivariate Normal Distribution at X with mean mu and a
    a cov that is a diagonal matrix. 
    Inputs: d    - dimension
            X    - dx1 to evaluate the PDF at
            mu   - dx1 vector (mean)
            sigd - dx1 vector (sqrt of diag of Sigma)
    """
    xSigSqrtinv = X_minus_mu/sigd
    logSqrtDetSigma = np.sum(np.log(sigd))
    quadform = np.sum(xSigSqrtinv**2, axis=0)
    ly = -0.5 * quadform - logSqrtDetSigma - 0.5 * d * np.log(2*np.pi)
    return ly