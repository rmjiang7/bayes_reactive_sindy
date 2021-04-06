data {
    int N; // Number of observations
    int M; // Number of species
    int D; // Number of possible reactions
    int D1; // Number of known rates
    matrix[M,D] stoichiometric_matrix;
    matrix[N,D] rate_matrix;
    vector[M - 1] y[N];

    row_vector[D1] known_rates;

    // horseshoe parameters
    real m0;
    real slab_scale;
    real slab_df;

    // optional parameters
    real<lower = 0> sigma;
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
    //real<lower = 0> tau_tilde;
    real alpha;

}

transformed parameters {
    matrix[N, M] rhs;
    row_vector[D] rates;
    real c2;
    real tau;
    vector[D - D1] lambda_tilde;

    {
        matrix[N,D] applied_rate_matrix;

        real tau0 = (m0 / (D - D1 - m0)) * (sigma / sqrt(N * M));
        tau = tau0; //* tau_tilde;

        c2 = slab_scale2 * c2_tilde;

        lambda_tilde = sqrt((c2 * square(lambda)) ./ (c2 + square(tau) * square(lambda)));

        if(D1 > 0) {
          rates[:D1] = to_row_vector(known_rates);
        }
        rates[D1 + 1:] = to_row_vector(tau * lambda_tilde .* unknown_rates_tilde);

        for(n in 1:N) {
          applied_rate_matrix[n, :] = rate_matrix[n,:] .* rates;
        }
        rhs = applied_rate_matrix * transpose(stoichiometric_matrix);
    }
}

model {
    unknown_rates_tilde ~ normal(0, 1);
    lambda ~ cauchy(0, 1);
    //tau_tilde ~ cauchy(0, 1);
    c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);

    for(i in 1:N) {
        y[i] ~ normal(rhs[i,1:M-1], noise_sigma);
    }
}

generated quantities {
    vector[M - 1] y_rep[N];
    real log_lik = 0;
    for(i in 1:N) {
        y_rep[i] = to_vector(normal_rng(rhs[i,1:M-1], noise_sigma));
        log_lik += normal_lpdf(y[i] | rhs[i,1:M-1], noise_sigma);
    }
}
