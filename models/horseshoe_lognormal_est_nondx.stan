data {
    int N; // Number of observations
    int M; // Number of species
    int D; // Number of possible reactions
    int D1; // Number of known rates

    real y[N + 1, M];
    real ts[N + 1];

    row_vector[D1] known_rates;

    // horsehoe parameters
    real m0;
    real slab_scale;
    real slab_df;

    // optional parameters
    real<lower = 0> sigma;
}

transformed data {
    real slab_scale2 = square(slab_scale);
    real half_slab_df = 0.5 * slab_df;
    real x_r[0];
    int x_i[0];
}

parameters {
    vector<lower = 0>[D - D1] unknown_rates_tilde;
    vector<lower = 0>[D - D1] lambda;
    real<lower = 0> c2_tilde;
    //real<lower = 0> tau_tilde;
    real alpha;

    real<lower = 0> noise_sigma[M];
}

transformed parameters {
    row_vector[D] rates;
    real c2;
    real tau;
    vector[D - D1] lambda_tilde;
    real y_hat[N, M];
    {
        real tau0 = (m0 / (D - D1 - m0)) * (sigma / sqrt(N * M));
        tau = tau0; //* tau_tilde;

        c2 = slab_scale2 * c2_tilde;

        lambda_tilde = sqrt((c2 * square(lambda)) ./ (c2 + square(tau) * square(lambda)));

        if(D1 > 0) {
          rates[:D1] = to_row_vector(known_rates);
        }
        rates[D1 + 1:] = to_row_vector(tau * lambda_tilde .* unknown_rates_tilde);
    }
    y_hat = integrate_ode_rk45(sys,
                               y[1,:],
                               ts[1],
                               ts[2:],
                               to_array_1d(rates),
                               x_r,x_i,
                               1e-6, 1e-5, 1e3);
}

model {
    // horseshoe priors
    unknown_rates_tilde ~ normal(0, 1);
    lambda ~ cauchy(0, 1);
    //tau_tilde ~ cauchy(0, 1);
    c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);

    noise_sigma ~ gamma(2, 2);

    for(i in 1:N) {
      y[i + 1,:] ~ lognormal(log(y_hat[i,:]), noise_sigma);
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
