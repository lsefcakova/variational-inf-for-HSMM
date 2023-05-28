from scipy.stats import poisson
import numpy as np 


''' function defined by jack with shifted poisson (gives 0  at state one i.e.)'''
# def h_j_pois(r : float, lambda_j: float):
#     h = poisson(mu = lambda_j)
#     if (1-h.cdf(x=r-2)) >= 1 : 
#         return 1
#     else:
#         return h.pmf(k=r-1)/(1-h.cdf(x=r-2))

# calculate hazard rates h_j(x)

def h_j_pois(r : float, lambda_j: float):
    h = poisson(mu = lambda_j)
    if (1-h.cdf(x=r-1)) >= 1 : 
        return 1
    else:
        return h.pmf(k=r)/(1-h.cdf(x=r-1))

def expand_matrix(T_hsmm, lambda_js_array , a_js_array):
    '''
    This function takes a valid transition 
    '''

    # check validity
    if not np.all(np.diagonal(T_hsmm) == 0):
        raise ValueError('This matrix is not valid. The diagonal entries of transition matrix (T_hsmm) must be all equal to 0.')
    # elif not np.all(T_hsmm.sum(axis = 1) == 1):
    #     raise ValueError('This matrix is not valid. The row sums of transition matrix (T_hsmm) must be equal to 1')

    T_expanded = np.zeros((sum(a_js_array),sum(a_js_array)))
    last_index = 0
    for state in range(T_hsmm.shape[1]):
        # build h vector
        h_j_vec = np.asarray([h_j_pois(r , lambda_js_array[state]) for r in range(1,a_js_array[state]+1)])
        # print(f'the 1- h vector is : {1 - h_j_vec}')
        # set diagonal matrix entries
        phi_jj = np.zeros((a_js_array[state],a_js_array[state]))
        np.fill_diagonal(phi_jj[:-1,1:], (1-h_j_vec[:-1])) #first to 2nd-to-last row and second to last column
        phi_jj[-1,-1] = (1-h_j_vec[-1])
        # print(f'the phi_jj is : {phi_jj}')

        # update phi_jj entries (diagonal matrices)
        T_expanded[last_index:last_index+a_js_array[state],last_index:last_index+a_js_array[state]] = phi_jj
        # update off diagonal matrices
        for state_k in range(T_hsmm.shape[1]):
            if state == state_k:
                pass # pass diagonal matrices (aleady filled)
            else:
                # calculate h_vec product with transition probability to next state scaling 
                phi_jk = h_j_vec * T_hsmm[state,state_k]
                # update matrix entries
                T_expanded[last_index:last_index+a_js_array[state] , sum(a_js_array[:state_k]) ] = phi_jk
        last_index = last_index + a_js_array[state] # add to index for jumps in loop
    return T_expanded

def test_matrix_expansion():
    print(f'This is an example of how conversion HSMM to HMM transition matrix is done with expand_matrix()')
    T3  = np.array([
                [0,0.8,0.2],
                [0.3,0, 0.7],
                [0.5,0.5,0]
                ])

    T_hsmm = T3.copy()
    print(f'The HSMM matrix is :')
    print(T_hsmm)

    a_js_array = np.array([2,2,2])
    lambda_js_array = np.array([1,2,3])

    print(f'a_js are: {a_js_array}')
    print(f'lambda_js are: {lambda_js_array}')

    a = expand_matrix(T_hsmm, lambda_js_array , a_js_array)
    print(f'sum of row entries :{a.sum(axis=1)}')
    print(a)