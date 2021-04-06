data {
    int N; // Number of observations
    int M; // Number of species
    int D; // Number of possible reactions
    int D1; // Number of known rates
    matrix[M,D] stoichiometric_matrix;
    matrix[N,D] rate_matrix;
    vector[M - 1] y[N];

    row_vector[D1] known_rates;
    real<lower = 0> noise_sigma;
}

parameters {
    row_vector<lower = 0>[D - D1] unknown_rates;
}

transformed parameters {
    matrix[N, M] rhs;
    row_vector[D] rates;

    if(D1 > 0) {
      rates[:D1] = to_row_vector(known_rates);
    }
    rates[D1 + 1:] = to_row_vector(unknown_rates);

    for(n in 1:N) {
        rhs[n,:] = to_row_vector(stoichiometric_matrix * to_vector(rate_matrix[n,:] .* rates));
    }
}

model {
    unknown_rates ~ normal(0, 1);
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
