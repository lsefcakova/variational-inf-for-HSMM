import numpy as np 
from pybasicbayes.distributions import Gaussian

'''
functions to helpgenerate data using a transition matrix used for simulations
'''

def generate_states(T: np.array, n_obs: int):

    states = [i for i in range(1,T.shape[0]+1)]
    state_seq = [np.random.randint(1,states[-1]+1)]
    for i in range(n_obs-1):
        state_seq.append(np.random.choice(states,p=T[state_seq[-1]-1]))
    
    ind_for_state_j = [np.where(np.asarray(state_seq)==j) for j in states]

    return np.asarray(state_seq,dtype=int), ind_for_state_j

def generate_data(emit : np.array, ind_for_state_j: np.array):
    obs = np.zeros((sum([len(j[0]) for j in ind_for_state_j]),emit[0].mu.shape[0]))
    for states in range(len(ind_for_state_j)):
        obs[ind_for_state_j[states]] = emit[states].rvs(size = len(ind_for_state_j[states][0]))
    return obs

def make_emissions(centers: list , kappa_0: float, nu_0:float,sigmas = [np.eye(2)]):
    emits = []
    if len(sigmas) == 1:
        sigmas = [np.eye(len(centers[0]))]*len(centers)
    elif len(sigmas)!= len(centers):
        raise ValueError('Wrong covariance matrix dimension')
    for i in range(len(centers)):
        emits.append(Gaussian(mu=np.array(centers[i]),
                    sigma=sigmas[i],
                    mu_0=np.zeros(len(centers[i])),
                    sigma_0=np.eye(len(centers)),
                    kappa_0=kappa_0,
                    nu_0=nu_0))
    return np.array(emits)


