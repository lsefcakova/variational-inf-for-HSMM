# hmmbase.py
import abc
import types

import numpy as np
import numpy.linalg as npl

# This is just for debugging
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

import pickle as pkl
import itertools

from matrix_expansion import *
from copy import deepcopy

from numpy import newaxis as npa
from scipy.special import digamma, gammaln

import scipy.spatial.distance as dist

import util

# This is for taking logs of things so we don't get -inf
eps = 1e-9


class VariationalHSMMBase(object, metaclass=abc.ABCMeta):
    """ Abstract base class for finite variational HMMs.  Provides the
        interface, basic structure and functions that all implementations will
        need.
    """

    # Interface

    @abc.abstractmethod
    def global_update():
        pass

    @abc.abstractmethod
    def infer():
        """ Perform inference. """
        pass

    @staticmethod
    def make_param_dict(prior_pi0, prior_A, prior_emit, mask=None):
        """ Given parameters make a dict that can be used to initialize an
            object.
        """
        return {'prior_pi0': prior_pi0, 'prior_A': prior_A,
                'prior_emit': prior_emit, 'mask': mask}

    def set_mask(self, mask):
        if mask is None:
            # All observations observed
            self.mask = np.zeros(self.obs.shape[0], dtype='bool')
        else:
            self.mask = mask.astype('bool')

    def __init__(self, obs, prior_pi0, prior_A, prior_emit, m_js,lambda_js, mask=None,
                 init_init=None, init_tran=None, verbose=False, sts=None):
        """
            obs : T x D np array of the observations in D dimensions (Can
                  be a vector if D = 1).

                         hyperparameters.

            prior_A : K x K np array containing the prior parameters
                          for the transition distributions. Use K dirichlet
                          hyperparameters (1 for each row).

            prior_emit : K x 1 np array containing the emission
                          distributions, these should be distributions from
                          pybasicbayes/distributions.py
                          
            m_js,lambda_js : K x 2 np.array governing the matrix expansion 

            mask : 1-d bool array of length T indicating which observations are
                   missing.  1 means missing.

            init_init : 1-d array of size K. Initial initial distribution.  If
                        None, then use the mean of prior_pi0.

            init_tran : 2-d array of size K x K.  Initial transition matrix.
                        If None, then use the row-means of prior_A.

            verbose : Default False.  Print out info while running.

            sts : 2d ndarray of length N.  True state sequence.
        """

        self.verbose = verbose

        self.sts = sts

        # Save the hyperparameters
        self.prior_pi0 = deepcopy(prior_pi0).astype('float64')

        self.prior_A = deepcopy(prior_A).astype('float64')
        self.prior_emit = deepcopy(prior_emit)
###
        # initialize sequences used for langrock and zuccini expansion
        self.m_js = deepcopy(m_js).astype('int')
        self.lambda_js = deepcopy(lambda_js).astype('int')
###

        # Initialize global variational distributions.
        self.var_pi0 = prior_pi0 / np.sum(prior_pi0)
        self.var_A = prior_A / np.sum(prior_A, axis=1)[:,np.newaxis]
        
        # We copy the prior objects becase the mean and covariance are the
        # initial values which can be set randomly when the object is created.
        self.var_emit = deepcopy(prior_emit)

        # Save the observations
        self.obs = obs
        self.set_mask(mask)

        # Number of states
        self.K = prior_A.shape[0]
        self.M = np.sum(self.m_js)

        # Initialize global variational distributions over the extended state space 

        # intermediate_mat = prior_A 
        # np.fill_diagonal(intermediate_mat, 0)
        # intermediate_mat = intermediate_mat / np.sum(intermediate_mat, axis=1)[:,np.newaxis]
        # self.var_B =  expand_matrix(intermediate_mat,self.lambda_js,self.m_js)

        self.var_pi0_ext = np.ones(self.M)/self.M
        self.prior_pi0_ext = np.ones(self.M)



        if obs.ndim == 1:
            self.T = obs.shape[0]
            self.D = 1
        elif obs.ndim == 2:
            self.T, self.D = obs.shape
        else:
            raise RuntimeError("obs must have 1 or 2 dimensions")

        self.elbo = -np.inf

    def lower_bound(self):
        """ Compute variational lower-bound.  This is approximate when
            stochastic optimization is used.
        """

        elbo = 0.

        # Initial distribution (only if more than one series, so ignore for now)
        p_pi = self.prior_pi0
        p_pisum = np.sum(p_pi)
        q_pi = self.var_pi0
        q_pidg = digamma(q_pi + eps)
        q_pisum = np.sum(q_pi)
        dg_q_pisum = digamma(q_pisum + eps)

        # Energy
        pi_energy = (gammaln(p_pisum + eps) - np.sum(gammaln(p_pi + eps))
                     + np.sum((p_pi-1.)*(q_pidg - dg_q_pisum)))
        # Entropy
        pi_entropy = -(gammaln(q_pisum + eps) - np.sum(gammaln(q_pi + eps))
                       + np.sum((q_pi-1.)*(q_pidg - dg_q_pisum)))

        # Transition matrix (each row is Dirichlet so can do like above)
        p_A = self.prior_A
        p_Asum = np.sum(p_A, axis=1)
        q_A = self.var_A
        q_Adg = digamma(q_A + eps)
        q_Asum = np.sum(q_A, axis=1)
        dg_q_Asum = digamma(q_Asum + eps)

        A_energy = (gammaln(p_Asum + eps) - np.sum(gammaln(p_A + eps), axis=1)
                    + np.sum((p_A-1)*(q_Adg - dg_q_Asum[:,npa]), axis=1))
        A_entropy = -(gammaln(q_Asum + eps) - np.sum(gammaln(q_A + eps), axis=1)
                     + np.sum((q_A-1)*(q_Adg - dg_q_Asum[:,npa]), axis=1))
        A_energy = np.sum(A_energy)
        A_entropy = np.sum(A_entropy)

        # Emission distributions -- does both energy and entropy
        emit_vlb = 0.
        for k in range(self.K):
            emit_vlb += self.var_emit[k].get_vlb()

        # Data term and entropy of states
        # This amounts to the sum of the logs of the normalization terms from
        # the forwards pass (see Beal's thesis).
        # Note: We use minus here b/c c_table is the inverse of \zeta_t in Beal.
        #lZ = -np.sum(np.log(self.c_table + eps))

        # We don't need the minus anymore b/c this is 1/ctable
        lZ = np.sum(np.logaddexp.reduce(self.lalpha_tilde, axis=1))

        elbo = (pi_energy + pi_entropy + A_energy + A_entropy
                + emit_vlb + lZ)

        return elbo

    def local_update(self, obs=None, mask=None):
        """ This is the local update for the batch version. Here we're creating
            modified parameters to run the forward-backward algorithm on to
            update the variational q distribution over the hidden states.

            These are always the same, and if we really need to change them
            we'll override the function.
        """
        if obs is None:
            obs = self.obs
        if mask is None:
            mask = self.mask


# this is a sequence of a,b a,b equations pairs going from hmm to extended hsmm formulation       # 
###
        # self.pi_tilde = digamma(self.var_pi0 + eps) - digamma(np.sum(self.var_pi0) + eps) # ??
        tran_sum = np.sum(self.var_A, axis=1)

        self.pi_tilde_ext = digamma(self.var_pi0_ext + eps) - digamma(np.sum(self.var_pi0_ext) + eps) # ??
        # tran_sum_ext = np.sum(self.var_B, axis=1)
###
###
        self.A_tilde = digamma(self.var_A + eps) - digamma(tran_sum[:,npa] + eps) #11?
        inter_matrix = self.A_tilde
        np.fill_diagonal(inter_matrix,0)
        self.B_tilde = expand_matrix(inter_matrix,self.lambda_js,self.m_js)
        # np.fill_diagonal(self.B_tilde,np.diagonal(self.var_B))
###
        # Compute log-likelihoods (only in hsmm space)
        for k, odist in enumerate(self.var_emit):
            self.lliks[:,k] = np.nan_to_num(odist.expected_log_likelihood(obs))

        # update forward, backward and scale coefficient tables
        self.forward_msgs() #12 done in log 
        self.backward_msgs() #13 done in log

###
        # self.var_x = self.lalpha + self.lbeta #14 in log self.var_x = q*(x_t=k)
        self.var_x_ext = self.lalpha_tilde + self.lbeta_tilde
###
###
        # self.var_x -= np.max(self.var_x, axis=1)[:,npa] 
        self.var_x_ext -= np.max(self.var_x_ext, axis=1)[:,npa] 
###
        #why do we substract the max value? is this step 15??
###
        # self.var_x = np.exp(self.var_x)
        self.var_x_ext = np.exp(self.var_x_ext)
###
        # exit log and convert to normal exponential value for normalization in line down
        # we will take the outer product to compute the probabilities kj in the global update 
        # that will update the transitions parameters 
###
        # self.var_x /= np.sum(self.var_x, axis=1)[:,npa]
        self.var_x_ext /= np.sum(self.var_x_ext, axis=1)[:,npa]
###     

        for i in range(len(self.m_js)):
            for k in range(self.K):
                if k == 0:
                    self.var_x[:,k] = np.sum(self.var_x_ext[:,0:self.m_js[k]],axis=1)
                else:
                    ind_u = np.sum(self.m_js[:k+1])
                    ind_l = ind_u - self.m_js[k]
                    self.var_x[:,k] = np.sum(self.var_x_ext[:,ind_l:ind_u],axis=1)


        self.var_x /= np.sum(self.var_x, axis=1)[:,npa]

        # normalizing step


    def forward_msgs(self, obs=None, mask=None):
        """ Creates an alpha table (matrix) where
            alpha_table[i,j] = alpha_{i}(z_{i} = j) = P(z_{i} = j | x_{1:i}).
            This also creates the scales stored in c_table. Here we're looking
            at the probability of being in state j and time i, and having
            observed the partial observation sequence form time 1 to i.

            obs : iterable of observation indices.  If None defaults to
                    all timestamps.

            See: http://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf
                 for an explanation of forward-backward with scaling.

            Override this for specialized behavior.
        """

        if obs is None:
            obs = self.obs
        if mask is None:
            mask = self.mask
###            
        # ltran = self.A_tilde
        ltran_ext = self.B_tilde
###
###
        ll = self.lliks
        ll_ext =  np.zeros((self.T,self.M))
        for i in range(self.T):
            row_i = []
            for j in range(self.K):
                row_i = row_i + self.m_js[j]*[ll[i][j]]
            ll_ext[i,:] = np.array(row_i)
###
###
        # lalpha = self.lalpha
        lalpha_tilde = self.lalpha_tilde
###
###
        # lalpha[0,:] = self.pi_tilde + ll[0,:] #log alpha ll-> pi from tutorial.pdf
        lalpha_tilde[0,:] = self.pi_tilde_ext + ll_ext[0,:] 
###
###
        for t in range(1,self.T):
            # self.lalpha[t] = np.logaddexp.reduce(lalpha[t-1] + ltran.T, axis=1) + ll[t]
            self.lalpha_tilde[t] = np.logaddexp.reduce(lalpha_tilde[t-1] + ltran_ext.T, axis=1) + ll_ext[t]

    def backward_msgs(self, obs=None, mask=None):
        """ Creates a beta table (matrix) where
            beta_table[i,j] = beta_{i}(z_{i} = j) = P(x_{i+1:T} | z_{t} = j).
            This also scales the probabilies. Here we're looking at the
            probability of observing the partial observation sequence from time
            i+1 to T given that we're in state j at time t.

            Override this for specialized behavior.
        """

        if obs is None:
            obs = self.obs
        if mask is None:
            mask = self.mask

###            
        # ltran = self.A_tilde
        ltran_ext = self.B_tilde
###
###
        ll = self.lliks
        ll_ext =  np.zeros((self.T,self.M))
        for i in range(self.T):
            row_i = []
            for j in range(self.K):
                row_i = row_i + self.m_js[j]*[ll[i][j]]
            ll_ext[i,:] = np.array(row_i)
###
###
        # lbeta = self.lbeta
        lbeta_tilde = self.lbeta_tilde
###
###
        # lbeta[self.T-1,:] = 0. #log of 1 
        self.lbeta_tilde[self.T-1,:] = 0. #log of 1 
###
###
        for t in range(self.T-2,-1,-1):
            # np.logaddexp.reduce(ltran + lbeta[t+1] + ll[t+1], axis=1,
                                # out=self.lbeta[t])
            np.logaddexp.reduce(ltran_ext + lbeta_tilde[t+1] + ll_ext[t+1], axis=1,
                                out=self.lbeta_tilde[t])
###

    def pred_logprob(self):
        """ Compute vector of predictive log-probabilities of data marked as
            missing in the `mask` member.

            Returns None if no missing data.
        """
        K = self.K
        obs = self.obs
        mask = self.mask
        nmiss = np.sum(mask)
        if nmiss == 0:
            return None

        logprob = np.zeros((nmiss,K))

        for k, odist in enumerate(self.var_emit):
            logprob[:,k] = np.log(self.var_x[mask,k]+eps) + odist.expected_log_likelihood(obs[mask,:])

        return np.mean(np.logaddexp.reduce(logprob, axis=1))

