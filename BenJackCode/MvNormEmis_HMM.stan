// HMM models
// with MVGaussian Emisison distributions
// Normal-Inverse-Wishart Priors

functions {
  
  /**  generate a N * K matrix with
  multivariate Gaussian emissions for t=1,..,N and j=1,..,K.
  */
  matrix log_mvnormEmissions(int N, int K, int D, vector[] y, vector[] mu, 
                          matrix[] Sigma) 
  {
    matrix[N, K] log_allprobs; 
    for (n in 1:N) {
      for (k in 1:K) {
        log_allprobs[n, k] = multi_normal_lpdf(y[n] |
                             mu[k], Sigma[k]);
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
  real llk_lp(int N, int K, int D, vector[] y, vector[] mu, matrix[] Sigma, vector[] gamma)
  {
    //matrix[N, K] emissions = exp(mvnormEmissions(N, K, D, y, mu, Sigma));
    matrix[N, K] log_emissions = log_mvnormEmissions(N, K, D, y, mu, Sigma);
    matrix[K, K] gamma_mat = gammaToMat(K, gamma);
    real llk =  forwardMessages(N, K, log_emissions, gamma_mat);
    return llk;
  }
  
}


data {
  int<lower=0> N; // length time series
  int<lower=0> D; // dimensionality
  int<lower=0> K; // number of states
  vector[D] y[N]; // data
  
  // hyperparms
  vector[D] mu_0; 
  real<lower=0> kappa_0; 
  real<lower=(D - 1)> nu_0; 
  cov_matrix[D] Psi_0;
  vector<lower=0>[K] alpha_0[K]; // prior dirichlet probs
}

parameters {
  //ordered[D] mu[K]; // mean gauss emission
  vector[D] mu[K]; // mean gauss emission
  cov_matrix[D] Sigma[K]; // correlation 
  simplex[K] gamma[K]; // transition prob mat 
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
  }
  // -- likelihood
  target += llk_lp(N, K, D, y, mu, Sigma, gamma); 
}

