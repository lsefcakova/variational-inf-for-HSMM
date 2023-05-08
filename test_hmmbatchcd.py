

import numpy as np
import matplotlib.pyplot as plt
import util
import hmmbatchcd as HMM

from scipy.spatial.distance import hamming as hd
from pybasicbayes.distributions import Gaussian
from util import *


def test_hmmbatchcd():
    """
    """

    K = 2
    D = 2
    kappa_0 = 1
    nu_0 = 4

    emit1 = Gaussian(mu=np.array([0,0]),
                     sigma=np.eye(2),
                     mu_0=np.zeros(2),
                     sigma_0=np.eye(2),
                     kappa_0=kappa_0,
                     nu_0=nu_0)
    emit2 = Gaussian(mu=np.array([5,5]),
                     sigma=np.eye(2),
                     mu_0=np.zeros(2),
                     sigma_0=np.eye(2),
                     kappa_0=kappa_0,
                     nu_0=nu_0)
    emit = np.array([emit1, emit2])

    N = 1000
    obs = np.array([emit[int(np.round(i/N))].rvs()[0]
                    for i in range(N)])
    
    mu_0 = np.zeros(D)
    sigma_0 = 0.75*np.cov(obs.T)
    kappa_0 = 0.1
    nu_0 = 4

    prior_emit = [Gaussian(mu_0=mu_0, sigma_0=sigma_0, kappa_0=kappa_0, 
                           nu_0=nu_0) for _ in range(K)]
    prior_emit = np.array(prior_emit)
    prior_tran = np.ones(K*K).reshape((K,K))
    prior_init = np.ones(K)

    hmm = HMM.VBHMM(obs, prior_init, prior_tran, prior_emit, maxit = 10**5,epsilon=10**(-8))
    hmm.infer()

    sts_true = np.array([int(np.round(i/N)) for i in range(N)])
    # hamming distance
    print('Hamming Distance = ', hmm.hamming_dist(hmm.var_x, sts_true)[0])

    # plot learned emissions over observations
    util.plot_emissions(obs, prior_emit, hmm.var_emit)
    plt.show()

    # plot elbo over iterations
    plt.plot(hmm.elbo_vec)
    plt.show()

    
if __name__ == '__main__':
    test_hmmbatchcd()
