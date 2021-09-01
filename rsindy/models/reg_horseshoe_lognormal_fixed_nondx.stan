data {
    int N; // Number of observations
    int M; // Number of species
    int M_obs; // Observed species
    int obs_idx[M_obs]; // Indices of observed speces

    int D; // Number of possible reactions
    int D1; // Number of known rates

    vector[M] y0;
    real y[N, M_obs];
    real ts[N + 1];

    vector[D1] known_rates;

    // horseshoe parameters
    real m0;
    real slab_scale;
    real slab_df;
    real<lower = 0> tau0;


    // noise model parameters
    real<lower = 0> noise_sigma;

}

transformed data {
    real slab_scale2 = square(slab_scale);
    real half_slab_df = 0.5 * slab_df;
}

parameters {
    vector<lower = 0>[D - D1] unknown_rates_tilde;
    vector<lower = 0>[D - D1] lambda;
    real<lower = 0> c2_tilde;
}

transformed parameters {
    vector[D] rates;
    real c2;
    real tau;
    vector[D - D1] lambda_tilde;
    vector[M] y_hat[N];
    {
        tau = tau0;

        c2 = slab_scale2 * c2_tilde;

        lambda_tilde = sqrt((c2 * square(lambda)) ./ (c2 + square(tau) * square(lambda)));

        if(D1 > 0) {
          rates[:D1] = known_rates;
        }
        rates[D1 + 1:] = tau * lambda_tilde .* unknown_rates_tilde;
    }
    y_hat = ode_rk45(sys,
                     y0,
                     ts[1],
                     ts[2:],
                     rates);
}

model {
    // horseshoe priors
    unknown_rates_tilde ~ normal(0, 1);
    lambda ~ cauchy(0, 1);
    c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);

    // model likelihood
    for(j in 1:M_obs) {
      y[ ,j] ~ lognormal(log(y_hat[ ,obs_idx[j]]), noise_sigma);
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
