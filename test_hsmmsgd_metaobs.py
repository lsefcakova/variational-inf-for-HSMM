

import numpy as np
import matplotlib.pyplot as plt
import util
import hsmmsgd_metaobs as HSMM

from scipy.spatial.distance import hamming as hd
from pybasicbayes.distributions import Gaussian
from util import *


from pybasicbayes import distributions as dist
from generate_data import *

from sklearn.cluster import KMeans


def test_hmmsgd_metaobs(maxit=50, metaobs_half=10, mb_sz=1):
    """
    """

    centers = np.array([np.array([0,0]),np.array([4,4]),np.array([0,6]),np.array([8,6])])

    K = 4           # number of states  
    D = 2        #have to be equal (symmetry)
    kappa_0 = 1.5     #??? variance of emissions ?
    nu_0 = 4        #??? smoothing?

    T = np.array([
        [0 ,0.3,0.6,0.1],
        [0.2, 0 ,0.7,0.1],
        [0.7,0.2, 0 ,0.1],
        [0.7,0.2,0.1, 0 ]
        ])

    N = 2000
    lambda_js = np.arange(1,K+1)*2 + 2


    emit = make_emissions(centers , kappa_0, nu_0)

    seq_hsmm, ind = generate_states_pois(T,lambda_js,N)



    obs_hsmm = generate_data(emit,ind)
	
    kmeans = KMeans(init="random",n_clusters=K,n_init=10,max_iter=300,random_state=1)
    kmeans.fit(obs_hsmm)

    mu_0 = kmeans.cluster_centers_ #np.zeros(D)
    sigma_0 = 0.75*np.cov(obs_hsmm.T)
    kappa_0 = 10 # high value for stability (low rescale when Gaussian mu_0 --> mu_mf resample(--> nat_to_standard()))
    # if we use the KMeans we are pretty confident in the centers so we can set a high value
    # consult this step with Jack and Benni !!!!!!!!!!!!!
    nu_0 = 4

    # define emissions for each super state
    prior_emit = [Gaussian(mu_0=mu_0[i], sigma_0=sigma_0, kappa_0=kappa_0, 
                            nu_0=nu_0) for i in range(K)]
    prior_emit = np.array(prior_emit)           # prior on emissions retype
    prior_A = np.ones(K*K).reshape((K,K))    # prior on transition matrix all 1
    np.fill_diagonal(prior_A, 0)

    prior_pi0 = np.ones(K)                     # ?
    # -----------------------------------------------------------------
    # page break
    m_js,lambda_js = np.ones(K)*10, np.array(lambda_js)
    
    
    # define object for inferrence from hmmbatchcd.py
    # # pass priors on init transitions and emissions for each state
    hsmm = HSMM.VBHSMM(obs_hsmm, prior_pi0, prior_A, prior_emit,m_js,lambda_js,maxit=maxit,metaobs_half=metaobs_half, mb_sz=mb_sz)  
    hsmm.infer()
    full_var_x = hsmm.full_local_update()

    # hamming distance
    print('Hamming Distance = ', hsmm.hamming_dist(full_var_x, seq_hsmm)[0])


    # plot learned emissions over observations

    # plots mean_field values (prior in green, posterior approximations in red)
    # prior values are green
    # var_emit are red 
    a = util.plot_emissions(obs_hsmm, prior_emit, hsmm.var_emit)
    plt.show()

    # plot elbo over iterations
    plt.plot(hsmm.elbo_vec)
    plt.show()

if __name__ == '__main__':
    test_hmmsgd_metaobs()
