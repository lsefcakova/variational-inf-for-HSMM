


import sys
import time
import numpy as np

# Just for debugging
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from copy import deepcopy

from numpy import newaxis as npa
from scipy.special import digamma, gammaln

from hsmmbase import VariationalHSMMBase
from pybasicbayes import distributions as dist

import util

# This is for taking logs of things so we don't get -inf
eps = 1e-9

tau0 = 1.
kappa0 = 0.7


class VBHSMM(VariationalHSMMBase):
    """ Batch coordinate-descent variational inference for hidden Markov
        models.

        obs : observations
        x : hidden states
        init : initial distribution (only useful for multiple series)
        tran : transition matrix
        emit : emission distributions

        The user passes in the hyperparameters for the initial, transition and
        emission distribution. We then store these as hyperparameters, and make
        copies of them to use as the variational parameters, these are the
        parameters we're doing updates on.

        The user should have each unique observation indexed in the emission
        hyperparameters and have those corresponding indexes listed in the
        observations. This way the user wont have to provide a map from the
        indexes to the observations, also it's a lot easier to deal with
        indexes than observations.
    """
    
    @staticmethod
    def make_param_dict(prior_pi0, prior_A, prior_emit,m_js,lambda_js, tau=tau0,
                        kappa=kappa0, mask=None):
        """ Given parameters make a dict that can be used to initialize an
            object.
        """
        return {'prior_pi0': prior_pi0, 'prior_A': prior_A,
                'prior_emit': prior_emit, 'm_js': m_js,
                'lambda_js': lambda_js, 'mask': mask, 'tau': tau,
                'kappa': kappa}

    def __init__(self, obs, prior_pi0, prior_A, prior_emit,m_js,lambda_js, mask=None,
                 tau=tau0, kappa=kappa0, init_init=None, init_tran=None, 
                 epsilon=1e-10, maxit=1000, batch_size=None, verbose=False, sts=None):
        """
            obs : T x D np array of the observations in D dimensions (Can
                  be a vector if D = 1).

            prior_pi0 : 1 x K np array containing the prior parameters
                         for the initial distribution.  Use Dirichlet
                         hyperparameters.

            prior_A : K x K np array containing the prior parameters
                          for the transition distributions. Use K dirichlet
                          hyperparameters (1 for each row).

            prior_emit : K x 1 np array containing the emission
                          distributions, these should be distributions from
                          pybasicbayes/distributions.py

            mask : 1-d bool array of length T indicating which observations are
                   missing.  1 means missing.

            init_init : 1-d array of size K. Initial initial distribution.  If
                        None, then use the mean of prior_pi0.

            init_tran : 2-d array of size K x K.  Initial transition matrix.
                        If None, then use the row-means of prior_A.

            epsilon : Threshold to decide algorithm has converged.  Default
                      1e-8.

            maxit : Maximum number of iterations to run optimization.
                    Default is 100.
        """

        super(VBHSMM, self).__init__(obs, prior_pi0, prior_A, prior_emit,
                                     m_js,lambda_js,
                                    mask=mask, init_init=init_init,
                                    init_tran=init_tran, verbose=verbose,
                                    sts=sts)
        
        self.batch = self.obs

        self.elbo = -np.inf
        self.tau = tau
        self.kappa = kappa
        self.lrate = tau**(-kappa)  # (t + tau)^{-kappa}
        
        self.epsilon = epsilon
        self.maxit = maxit

        # Need to do this in the derived class because the sizes are different
        # depending on if we're batch or stochastic.
        # var_x is a T x K matrix so the k'th column represents the k'th
        # state over all T observations. So q[t,k] would represent the weight
        # for the k'th state at time t.
        self.var_x = np.random.rand(self.T, self.K)
        self.var_x /= np.sum(self.var_x, axis=1)[:,np.newaxis]

        # These are all members so we don't have to reinitialize them every
        # time we call the forward and backward functions
        #self.alpha_table = np.zeros((self.T, self.K))
        #self.beta_table = np.zeros((self.T, self.K))
        #self.c_table = np.zeros(self.T)
        
        self.alpha_table = np.zeros((self.T, self.M))
        self.beta_table = np.zeros((self.T, self.M))
        self.c_table = np.zeros(self.T)

        # self.lalpha = np.empty((self.T, self.K))
        # self.lbeta = np.empty((self.T, self.K))
        self.lliks = np.empty((self.T, self.K))

        self.lalpha_tilde = np.empty((self.T, self.M))
        self.lbeta_tilde = np.empty((self.T, self.M))
        self.lliks_tilde = np.empty((self.T, self.M))

        # The modified parameters used in the local update
        self.pi_tilde = np.zeros(self.K)
        self.A_tilde = np.zeros((self.K, self.K))

        self.pi_tilde_extended = np.zeros(self.M)

        # the variable to save the extended space log q(x)
        self.var_x_ext = np.empty((self.T, self.M))

        # Stuff for viterbi that we don't use.  It could go in hmmbase.py
        #self.a_table = np.zeros((self.T, self.K))
        #self.viterbi_table = np.zeros((self.T, self.K))
        
    def infer(self):
        """ Runs stochastic variational inference algorithm. This works with
            only a subset of the data.
        """

        self.obs_full = self.obs.copy()
        self.obs[self.mask,:] = np.nan

        epsilon = self.epsilon
        maxit = self.maxit

        self.elbo_vec = np.inf*np.ones(maxit)
        self.pred_logprob_mean = np.nan*np.ones(maxit)
        self.pred_logprob_std = np.nan*np.ones(maxit)

        self.iter_time = np.nan*np.ones(maxit)

        for it in range(maxit):
			
            start_time = time.time()

            # (t + tau)^{-kappa}
            self.lrate = (it + self.tau)**(-self.kappa)

            self.local_update()
            #self.global_update()
            self.global_update()
			
            self.iter_time[it] = time.time() - start_time

            # Keep getting matrix not positive definite in lower_bound
            # function
            lb = self.lower_bound()

            if self.verbose:
                print("iter: %d, ELBO: %.2f" % (it, lb))
                sys.stdout.flush()
			
            if np.abs(lb - self.elbo) <= epsilon : #np.allclose(lb, self.elbo, rtol=epsilon):
                print(np.abs(lb - self.elbo) <= epsilon)
                print(f'terminated early - convergence, \n elbo : {self.elbo} \n lower bound : {lb}', )
                print(f'allclose  = {np.allclose(lb, self.elbo, rtol=epsilon)}, diff : {self.elbo - lb}')
                break
            else:
                self.elbo = lb
                self.elbo_vec[it] = lb
                tmp = self.pred_logprob()
                if tmp is not None:
                    self.pred_logprob_mean[it] = np.mean(tmp)
                    self.pred_logprob_std[it] = np.std(tmp)

        lbidx = np.where(np.logical_not(np.isinf(self.elbo_vec)))[0]
        self.elbo_vec = self.elbo_vec[lbidx]
        self.pred_logprob_mean = self.pred_logprob_mean[lbidx]
        self.pred_logprob_std = self.pred_logprob_std[lbidx]
        self.iter_time = self.iter_time[lbidx]

        # Save Hamming distance
        if self.sts is not None:
            self.hamming, self.perm = self.hamming_dist(self.var_x, self.sts)

        self.obs = self.obs_full

    def global_update(self):
        """ Perform global updates based on batch following the stochastic
            natural gradient.
        """
        
        """ Perform global updates based on batch following the stochastic
            natural gradient.
        """
        

        batch = self.obs
        mask = self.mask
        
        inds = np.logical_not(mask)
        
        lrate = self.lrate
        #batchfactor = self.batchfactor

        # Perform stochastic gradient update on global params.

        # Initial state distribution
        self.var_pi0_ext = self.prior_pi0_ext + self.var_x_ext[0,:]
        for i in range(len(self.m_js)):
            for k in range(self.K):
                if k == 0:
                    self.var_pi0[k] = np.sum(self.var_pi0_ext[0:self.m_js[k]])
                else:
                    ind_u = np.sum(self.m_js[:k+1])
                    ind_l = ind_u - self.m_js[k]
                    self.var_pi0[k] = np.sum(self.var_pi0_ext[ind_l:ind_u])
                    
                    ##here
        
        #######
        
        # Transition distribution
        # Convert to natural parameters
        nats_old = self.var_A - 1.

        # Mean-field update
        tran_mf = self.prior_A.copy()
        for t in range(1, self.T):
            tran_mf += np.outer(self.var_x[t-1,:], self.var_x[t,:]) 

        # Convert result to natural params
        nats_t = tran_mf - 1.

        # Perform update according to stochastic gradient
        # (Hoffman, pg. 17)
        nats_new = (1.-lrate)*nats_old + lrate*nats_t

        # Convert results back to moment params
        self.var_A = nats_new + 1.
        
        #######

        # Emission distributions
        #inds = np.logical_not(self.mask)
        for k in range(self.K):
            G = self.var_emit[k]
            
            #nats_old = np.array([G.natural_hypparam])

            # Do mean-field update for this component
            #mu_mf, sigma_mf, kappa_mf, nu_mf = \
            #        util.NIW_meanfield(G, batch[inds,:], self.var_x[inds,k])
            
            #G.meanfieldupdate(batch[inds,:], self.var_x[inds,k])

            #nats_t = np.array([mu_mf, sigma_mf, kappa_mf, nu_mf])
			
            # Convert to natural parameters
            #nats_t = util.NIW_mf_natural_pars(mu_mf, sigma_mf,
            #                                  kappa_mf, nu_mf)

            # Convert current estimates to natural parameters
            #nats_old = util.NIW_mf_natural_pars(G.mu_mf, G.sigma_mf,
            #                                    G.kappa_mf, G.nu_mf)
			
            # Perform update according to stochastic gradient
            # (Hoffman, pg. 17)
            #nats_new = (1.-lrate)*nats_old + lrate*nats_t
            #print(nats_new) ##works!! does it even mean anything?
            #print("--------")
            #G.natural_hypparam=nats_new
            
            G.meanfield_sgdstep(batch[inds,:], self.var_x[inds,k], 1, lrate)
            
            # Convert new params into moment form and store back in G
            #util.NIW_mf_moment_pars(G, *nats_new)
