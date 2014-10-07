"""
The Circular Law proved by Terence Tao and Van Vu considers the eigenvalues of a 
random matrix A_N with independent and identically distributed, complex-valued
random variables A_{ij} for 1 <= i,j <= N that satisfy the following:

mean = E(A_{11}) = 0
variance = E(A_{11}^2) - E(A_{11})^2 = 1,

where E is the mathematical expectation. As N approaches infinity, the 
eigenvalues of the scaled matrix

(1/\sqrt{N}) * A_N

form a uniform distribution on the unit disk in the complex plane. More precisely,
the Empirical Spectral Distribution (ESD) is defined as 

\mu_A(x,y):= (1/n) |{1<=i <= n, Re \lam_i <=x, Im\lam_i <= y}|

where \lam_i, for i=1,...,n are the eigenvalues of a matrix A. The limiting
distribution of the scaled ESD converges to a distribution independent of the 
choice of entries, therefore proving a universality principle [1].

Appendix: After seeing a paper by Hoi Nguyen and Sean O'Rourke [2], I noticed a 
scaling possibility between the classic Wigner's semi-circle law [3] and
Tao and Vu's circular law. Although Nguyen and O'Rourke consider correleated 
entries, I consider a band of independent entires (for Wigner, this is the 
the diagonal, for Tao-Vu, it's the entire matrix). In the code, a parameter 
t in [0,1] scales between these two cases.

[1] Random matrices: Universality of ESDs and the circular law
http://arxiv.org/abs/0807.4898

[2] The Elliptic Law
http://arxiv.org/abs/1208.5883

[3] A blogpost by Terence Tao on the semi-circle law
https://terrytao.wordpress.com/tag/wigner-semi-circular-law/

The code below chooses complex Gaussian random variables as a distribution.

Ricky Kwok rickyk9487@gmail.com 2014-10-07
"""

import random
import numpy as np
import matplotlib.pyplot as plt

def get_RM(N,t):
    """ Produces a random matrix of size N with Gaussian entries and 
        returns the eigenvalues of the scaled matrix."""
    dim = (N,N)
    var = 1.0/np.sqrt(2.0)
    H = np.zeros(dim, dtype = complex) 
    # initialize the complex-type matrix
    for i in range (N):
        H[i,i] = random.normalvariate(0,1)
        # the diagonal is real, normal zero one
        for j in range(N):
            if abs(j-i) <= np.floor(t*N) and j > i:
                # within a distance of the diagonal 
                R = random.normalvariate(0,var)
                I = random.normalvariate(0,var)
                H[i,j] = complex(R,I)
                H[j,i] = H[i,j].conjugate()
            elif abs(j-i) > np.floor(t*N):
                # the top right and bottom left
                H[i,j] = random.normalvariate(0,1)
                
    scale = np.sqrt(N ** -1)
    W = scale * H
    lam = np.linalg.eigvals(W)
    return lam

def plot_eig(lam, t):
    """ Given an array lam of floats (complex), plots the values in a histogram
        (scatterplot). """
    plt.close()
    if t >= 1.0:
        # at t = 1.0, all eigenvalues become real and we get Wigner's law
        plt.hist(lam)
    else:
        plt.scatter(lam.real,lam.imag) 
    plt.show()

def main():
    N = 500
    # size of matrix. At 5000 we begin to see clear patterns
    t = 1.0 
    # parameter between 0 and 1. (t*N) diagonals will be unitary
    # for t = 1, the entire matrix is unitary
    # for t = 0, the entire matrix has independent entries
    
    lam = get_RM(N, t)
    plot_eig(lam, t)
    
if __name__ == "__main__":
    main()
    