

import numpy as np
import matplotlib.pyplot as plt
import util
import hmmbatchcd as HMM

from scipy.spatial.distance import hamming as hd
from pybasicbayes.distributions import Gaussian
from util import *

from pybasicbayes import distributions as dist
from generate_data import *

from sklearn.cluster import KMeans

def test_hmmbatchcd():
    """
    """

    centers = np.array([np.array([0,0]),np.array([6,0]),np.array([4,4]),np.array([0,6]),np.array([12,8])])
    # centers = [np.array([0,0]),np.array([6,6])]
    # centers = [np.array([0,0,0]),np.array([6,0,6]),np.array([4,4,4])]

    K = 5           # number of states  
    D = 2        #have to be equal (symmetry)
    kappa_0 = 0.5     #??? variance of emissions ?
    nu_0 = 4        #??? smoothing?

    T = np.array([
        [0.89,0.01,0.01,0.05,0.04],
        [0.05,0.8,0.05,0.05,0.05],
        [0.05,0.1,0.7,0.05,0.1],
        [0.1,0.1,0.05,0.7,0.05],
        [0.025,0.025,0.025,0.025,0.9]
        ])
    # T = np.array([[0.8,0.2],[0.3,0.7]])

    N = 3000

    emit = make_emissions(centers , kappa_0, nu_0)
    seq, ind = generate_states(T,N)

    obs = generate_data(emit,ind)
    
    mu_0 = np.zeros(D)
    sigma_0 = 0.75*np.cov(obs.T)
    kappa_0 = 0.1
    nu_0 = 4

    kmeans = KMeans(init="random",n_clusters=K,n_init=10,max_iter=300,random_state=42)
    kmeans.fit(obs)

    mu_0 = kmeans.cluster_centers_ #np.zeros(D)
    sigma_0 = 0.75*np.cov(obs.T)
    kappa_0 = 10 # high value for stability (low rescale when Gaussian mu_0 --> mu_mf resample(--> nat_to_standard()))
    # if we use the KMeans we are pretty confident in the centers so we can set a high value
    # consult this step with Jack and Benni !!!!!!!!!!!!!
    nu_0 = 4


    # define emisssion RV for each state (2)
    prior_emit = [Gaussian(mu_0=mu_0[i], sigma_0=sigma_0, kappa_0=kappa_0, 
                        nu_0=nu_0) for i in range(K)]
    prior_emit = np.array(prior_emit)           # prior on emissions retype
    prior_tran = np.ones(K*K).reshape((K,K))    # prior on transition matrix all 1
    prior_init = np.ones(K)

    # define object for inferrence from hmmbatchcd.py
    # pass priors on init transitions and emissions for each state
    hmm = HMM.VBHMM(obs, prior_init, prior_tran, prior_emit,maxit=1000, epsilon=10**(-12))  
    hmm.infer() # do inference 
    sts_true = seq # true states 
    # hamming distance
    print(prior_emit[0].mu_mf)
    print('Hamming Distance = ', hmm.hamming_dist(hmm.var_x, sts_true)[0]) 

    # plot learned emissions over observations

    # plots mean_field values (prior in green, posterior approximations in red)
    # prior values are green
    # var_emit are red 
    a = util.plot_emissions(obs, prior_emit, hmm.var_emit)
    plt.show()

    # plot elbo over iterations
    plt.plot(hmm.elbo_vec)
    plt.show()

    
if __name__ == '__main__':
    test_hmmbatchcd()
