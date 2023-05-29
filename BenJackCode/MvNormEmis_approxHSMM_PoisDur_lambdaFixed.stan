// approx HsMM models
// with Poisson duration distribution
// with MVGaussian Emisison distributions
// Normal-Inverse-Wishart Priors

functions {
  
  real c_hazard_dwell_poisson(int r, real lambda) 
  {
    // shifted 
    return exp(poisson_lpmf(r-1 | lambda))/(1-poisson_cdf(r-2, lambda));
  }
  
  /** B_ii poiss
  */
  matrix B_ii_poisson(int m_i, real lambda_i) 
  {
      matrix[m_i, m_i] B_ii;
      B_ii = rep_matrix(0, m_i, m_i);
      for( i in 1:(m_i-1)){
          B_ii[i, i+1] = 1 - c_hazard_dwell_poisson(i, lambda_i);
      }
      B_ii[m_i, m_i] = 1-c_hazard_dwell_poisson(m_i, lambda_i);
      return B_ii;
  }
  
  /** B_ij  poiss
  */
  matrix B_ij_poisson(int m_i, int m_j, real a_ij, real lambda_i)
  {
      matrix[m_i, m_j] B_ij;
      B_ij = rep_matrix(0, m_i, m_j);
      for( i in 1:m_i){
        B_ij[i, 1] =  a_ij * c_hazard_dwell_poisson(i, lambda_i);
      }
      return B_ij;
  }
  
  /** B_matrix_poisson;
  */
  matrix B_matrix_poisson(int K, int[] m_vect, matrix a_mat, vector lambda_vec)
  {
      int sum_m = sum(m_vect);
      matrix[sum_m, sum_m] B; 
      int m_vect_temp[K+1];
      B = rep_matrix(0, sum_m, sum_m);
      m_vect_temp[1] = 0;
      for(i in 1:K){
        m_vect_temp[i+1] = m_vect[i];//Adding a 0 to this makes the indexing below easier
      }
      for(i in 1:K){
        for(j in 1:K){
          if(i ==j){
            B[(sum(m_vect_temp[1:i])+1):sum(m_vect_temp[1:(i+1)]), 
            (sum(m_vect_temp[1:j])+1):sum(m_vect_temp[1:(j+1)])] =
             B_ii_poisson(m_vect_temp[i+1], lambda_vec[i]);
          } 
          else{
            B[(sum(m_vect_temp[1:i])+1):sum(m_vect_temp[1:(i+1)]), 
            (sum(m_vect_temp[1:j])+1):sum(m_vect_temp[1:(j+1)])] =
             B_ij_poisson(m_vect_temp[i+1], m_vect_temp[j+1], a_mat[i,j], lambda_vec[i]);
          }
        }
      }
      return B;
    }
    
  /** convert (K-1) simplex gamma to a K*K matrix transition matrix.
  */
  matrix a_to_mat(int K, vector[] a)  
  {
    matrix[K, K] A = rep_matrix(0, K, K); 
    int count_j = 0;
    for (i in 1:K) {
      count_j = 0;
      for (j in 1:K) {
        if(i != j) {
          count_j += 1;
          A[i, j] = a[i][count_j]; 
        }
      }
    }
    return A;
  }
  
  /**  generate a N * K matrix with
  multivariate Gaussian emissions for t=1,..,N and j=1,..,K.
  */
  matrix log_mvnormEmissions(int N, int K, int D, int[] m, vector[] y, vector[] mu, 
                          matrix[] Sigma) 
  {
    int sum_m = sum(m);
    matrix[N, sum_m] log_allprobs; 
    
    for (n in 1:N) {
      for (k in 1:K) {
        if(k ==1) {
          for (i in 1:m[k]) {
            log_allprobs[n, i] = multi_normal_lpdf(y[n] |
                             mu[k], Sigma[k]);
          }
        }
        else {
          for(i in 1:m[k]) {
            log_allprobs[n, sum(m[1:(k-1)]) + i] = multi_normal_lpdf(y[n] |
                             mu[k], Sigma[k]);
          }
        }
      }
    }
    return log_allprobs; 
  }
  
  
  /** perform forward algorithm to 
  generate alpha_t (j) for t=1,..,N and j=1,.., M = sum(m)
  via forward dynamic programming - p.243 (Zucchini et al.).
  */
  // FASTER BUT NOT NUMERICALLY STABLE 
  /*
  real forwardMessages(int N, int K, matrix emissions, matrix gamma_mat)
  {
    vector[K] foo;
    real sumfoo;
    real lscale;
    // alpha_1
    for (k in 1:K) {
      foo[k] = emissions[1, k];
    }
    sumfoo = sum(foo);
    lscale = log(sumfoo);
    foo = foo/sumfoo;
    // alpha_t, t = 2, ..., N
    for (i in 2:N) {
      foo = (foo'*gamma_mat .* emissions[i, :])';
      sumfoo = sum(foo);
      lscale = lscale + log(sumfoo);
      foo = foo/sumfoo;
    }
    return lscale;
  }
  */
  
  // SLOWER BUT NUMERICALLY STABLE 
  real forwardMessages(int N, int K, matrix log_emissions, matrix gamma_mat)
  {
    vector[K] log_foo;
    vector[K] log_foo_temp;
    real log_sumfoo;
    real lscale;
    // alpha_1
    for (k in 1:K) {
      log_foo_temp[k] = log_emissions[1, k];
    }
    log_sumfoo = log_sum_exp(log_foo_temp);
    lscale = log_sumfoo;
    log_foo = log_foo_temp - log_sumfoo;
    
    // alpha_t, t = 2, ..., N
    for (i in 2:N) {
      //log_foo_temp = (log(exp(log_foo)'*gamma_mat) + log_emissions[i, :])';
      for(k in 1:K){
        log_foo_temp[k] = log_sum_exp(log_foo + log(gamma_mat[:, k])) + log_emissions[i, k];
      }
      log_sumfoo = log_sum_exp(log_foo_temp);
      lscale += log_sumfoo;
      log_foo = log_foo_temp - log_sumfoo;
    }
    return lscale;
  }
  
  /** convert simplex gamma to a K*K matrix. 
  */
  matrix gammaToMat(int K, vector[] gamma)
  {
    matrix[K, K] gamma_mat;
    // Creating Gamma Matrix for Forward
    for(i in 1:K)
      for(j in 1:K)
        gamma_mat[i, j] = gamma[i][j];
    return gamma_mat;
  }

  /** compute likelihood p(y_1, .., y_T  | )
  */
  real llk_lp(int N, int K, int D, int[] m, vector[] y, vector[] mu, matrix[] Sigma, vector lambda, vector[] a)
  {
    int M = sum(m);
    matrix[K, K] A =  a_to_mat(K, a);
    matrix[M, M] B = B_matrix_poisson(K, m, A, lambda);
    //matrix[N, K] emissions = exp(mvnormEmissions(N, K, D, y, mu, Sigma));
    matrix[N, M] log_emissions = log_mvnormEmissions(N, K, D, m, y, mu, Sigma);
    real llk =  forwardMessages(N, M, log_emissions, B);
    return llk;
  }
  
}


data {
  int<lower=0> N; // length time series
  int<lower=0> D; // dimensionality
  int<lower=0> K; // number of states
  vector[D] y[N]; // data
  int m[K]; // size state aggr
  
  // hyperparms
  vector[D] mu_0; 
  real<lower=0> kappa_0; 
  real<lower=(D - 1)> nu_0; 
  cov_matrix[D] Psi_0;
  vector<lower=0>[K-1] alpha_0[K]; // prior dirichlet probs
  // Dwell hyperparams
  //real<lower=0> a_0[K]; // rate pois dwell
  //real<lower=0> b_0[K]; //   "" 
  vector<lower=0>[K] lambda;
}

parameters {
  //ordered[D] mu[K]; // mean gauss emission
  vector[D] mu[K]; // mean gauss emission
  cov_matrix[D] Sigma[K]; // correlation 
  simplex[K-1] gamma[K]; // transition prob mat 
  //vector<lower=0>[K] lambda;
}


model {
  // -- priors
  for (k in 1:K) {
    // Sigma
    target += inv_wishart_lpdf(Sigma[k] | nu_0, Psi_0);
    // mu
    target += multi_normal_lpdf(mu[k] | mu_0, 1/kappa_0*Sigma[k]);
    // gamma
    target += dirichlet_lpdf(gamma[k] | alpha_0[k]);
    // lambda            
    //target += gamma_lpdf(lambda[k] | a_0[k], b_0[k]);
  }
  // -- likelihood
  target += llk_lp(N, K, D, m, y, mu, Sigma, lambda, gamma); 
}

