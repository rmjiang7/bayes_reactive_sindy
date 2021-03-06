data {
    int N; // Number of observations
    int M; // Number of species
    int M_obs; // Observed species
    int obs_idx[M_obs]; // Indices of observed speces

    int D; // Number of possible reactions
    int D1; // Number of known rates

    real y0[M];
    real y[N, M_obs];
    real ts[N + 1];

    vector[D1] known_rates;

    // horsehoe parameters
    real m0;
    real slab_scale;
    real slab_df;
    real<lower = 0> sigma;

    // noise model parameters
    real<lower = 0> noise_sigma;

}

parameters {
    vector<lower = 0>[D - D1] unknown_rates_tilde;
    vector<lower = 0>[D - D1] lambda;
    //real<lower = 0> c2_tilde;
    real<lower = 0> tau_tilde;
}

transformed parameters {
    vector[D] rates;
    real tau;
    vector[M] y_hat[N];
    {
        tau = sigma * tau_tilde;

        lambda_tilde = lambda;

        if(D1 > 0) {
          rates[:D1] = known_rates;
        }
        rates[D1 + 1:] = tau * lambda .* unknown_rates_tilde;
    }
    y_hat = ode_rk45(sys,
                     to_vector(y0),
                     ts[1],
                     ts[2:],
                     rates);x
}

model {
    // horseshoe priors
    unknown_rates_tilde ~ normal(0, 1);
    lambda ~ cauchy(0, 1);
    tau_tilde ~ cauchy(0, 1);

    // model likelihood
    for(i in 1:N) {
      for(j in 1:M_obs) {
        y[i,j] ~ lognormal(log(y_hat[i,obs_idx[j]]), noise_sigma);
      }
    }
}

generated quantities {
  real y_rep[N, M];

  for(i in 1:N) {
    for(j in 1:M) {
      y_rep[i,j] = lognormal_rng(log(y_hat[i,j]), noise_sigma);
    }
  }

}
