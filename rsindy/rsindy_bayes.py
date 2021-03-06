import numpy as np
from cmdstanpy import CmdStanModel
from rsindy.rsindy import RSINDy
import os
import tempfile


class RSINDyNonRegularized(RSINDy):

    def _fit_dx(self,
                X_obs,
                ts,
                S,
                R,
                known_rates=[],
                fit_params={},
                model_params={}):

        y = []
        for i in range(1, X_obs.shape[0]):
            y.append((X_obs[i, :] - X_obs[i - 1, :]) / (ts[i] - ts[i - 1]))

        X_obs = X_obs[:-1]
        extended_X_obs = np.hstack([X_obs, np.ones((X_obs.shape[0], 1))])
        masked_X_obs = []
        for i in range(X_obs.shape[0]):
            ap = extended_X_obs[i, :] * (R == 1)
            ap += (extended_X_obs[i, :] * (R == 2)) ** 2
            ap_mask = ap + (ap == 0).astype(np.float32)
            masked_X_obs.append(np.prod(ap_mask, axis=1))
        pre_applied_stoichiometries = np.vstack(masked_X_obs)

        data = {
            'N': pre_applied_stoichiometries.shape[0],
            'M': S.T.shape[0],
            'D': S.T.shape[1],
            'D1': len(known_rates),
            'stoichiometric_matrix': S.T,
            'rate_matrix': pre_applied_stoichiometries,
            'y': np.array(y),
            'known_rates': known_rates
        }

        default_model_params = {'noise_sigma': 1}
        default_model_params = {**default_model_params, **model_params}
        data = {**data, **default_model_params}

        if default_model_params['noise_sigma'] <= 0:
            file = "models/regression_normal_est_dx.stan"
        else:
            file = "models/regression_normal_fixed_dx.stan"
        file = os.path.join(os.path.dirname(__file__), file)

        model = CmdStanModel(stan_file=file)

        default_fit_params = {'chains': 4,
                              'iter_warmup': 1000,
                              'iter_sampling': 1000,
                              'optimize': False,
                              'init': None,
                              'show_progress': False,
                              'variational': False,
                              'algorithm': 'meanfield',
                              'v_iters': 1000,
                              'v_grad_samples': None,
                              'v_elbo_samples': None}
        default_fit_params = {**default_fit_params, **fit_params}

        if default_fit_params['optimize']:
            fit = model.optimize(data=data)
        elif default_fit_params['variational']:
            fit = model.variational(
                data=data,
                iter=default_fit_params['v_iters'],
                grad_samples=default_fit_params['v_grad_samples'],
                elbo_samples=default_fit_params['v_elbo_samples'],
                algorithm=default_fit_params['algorithm'],
                require_converged=False)
        else:
            fit = model.sample(
                data=data,
                chains=default_fit_params['chains'],
                iter_warmup=default_fit_params['iter_warmup'],
                iter_sampling=default_fit_params['iter_sampling'],
                inits=default_fit_params['init'],
                refresh=1,
                show_progress=default_fit_params['show_progress'])

        return fit

    def _fit_non_dx(self,
                    X_obs,
                    ts,
                    S,
                    R,
                    known_rates,
                    fit_params,
                    model_params):
        raise NotImplementedError


class RSINDyRegularizedHorseshoe(RSINDy):

    def _fit_dx(self,
                X_obs,
                ts,
                S,
                R,
                known_rates=[],
                fit_params={},
                model_params={}):

        # Estimate Derivatives using finite differences
        y = []
        for i in range(1, X_obs.shape[0]):
            y.append((X_obs[i, :] - X_obs[i - 1, :]) / (ts[i] - ts[i - 1]))
        y = np.vstack(y)

        X_obs = X_obs[:-1]
        extended_X_obs = np.hstack([X_obs, np.ones((X_obs.shape[0], 1))])
        masked_X_obs = []
        for i in range(X_obs.shape[0]):
            ap = extended_X_obs[i, :] * (R == 1)
            ap += (extended_X_obs[i, :] * (R == 2)) ** 2
            ap_mask = ap + (ap == 0).astype(np.float32)
            masked_X_obs.append(np.prod(ap_mask, axis=1))
        pre_applied_stoichiometries = np.vstack(masked_X_obs)

        data = {
            'N': pre_applied_stoichiometries.shape[0],
            'M': S.T.shape[0],
            'D': S.T.shape[1],
            'D1': len(known_rates),
            'stoichiometric_matrix': S.T,
            'rate_matrix': pre_applied_stoichiometries,
            'y': y,
            'known_rates': known_rates
        }

        default_model_params = {'m0': 10,
                                'slab_scale': 1,
                                'slab_df': 2,
                                'sigma': 1,
                                'noise_sigma': 1}
        default_model_params = {**default_model_params, **model_params}
        data = {**data, **default_model_params}

        if default_model_params['noise_sigma'] <= 0:
            file = "models/horseshoe_normal_est_dx.stan"
        else:
            file = "models/horseshoe_normal_fixed_dx.stan"
        file = os.path.join(os.path.dirname(__file__), file)

        model = CmdStanModel(stan_file=file)

        default_fit_params = {'chains': 4,
                              'iter_warmup': 1000,
                              'iter_sampling': 1000,
                              'optimize': False,
                              'init': None,
                              'show_progress': False,
                              'variational': False,
                              'algorithm' : 'meanfield',
                              'v_iters': 1000,
                              'v_grad_samples': None,
                              'v_elbo_samples': None}
        default_fit_params = {**default_fit_params, **fit_params}

        if default_fit_params['optimize']:
            fit = model.optimize(data=data,
                                 inits=default_fit_params['init'])
        elif default_fit_params['variational']:
            fit = model.variational(
                data=data,
                iter=default_fit_params['v_iters'],
                grad_samples=default_fit_params['v_grad_samples'],
                elbo_samples=default_fit_params['v_elbo_samples'],
                algorithm=default_fit_params['algorithm'],
                require_converged=False)
            fit.variational_sample.columns = fit.column_names
        else:
            fit = model.sample(
                data=data,
                chains=default_fit_params['chains'],
                iter_warmup=default_fit_params['iter_warmup'],
                iter_sampling=default_fit_params['iter_sampling'],
                inits=default_fit_params['init'],
                refresh=1,
                show_progress=default_fit_params['show_progress'])

        return fit

    def _create_non_derivative_stan_model(self, S, R,
                                          regularized = True,
                                          additive = False,
                                          prior_predictive = False):
        model_str = """
        functions {{
            vector sys(real t,
                       vector y,
                       vector theta) {{
                vector[{}] dydt;
                vector[{}] v;
                matrix[{}, {}] S = {};
                {}
                dydt = S * v;
                return dydt;
            }}
        }}
        """
        if regularized:
            if prior_predictive:
                base_file = "models/reg_horseshoe_lognormal_fixed_nondx_pr_pred.stan"
            elif additive:
                base_file = "models/reg_horseshoe_lognormal_fixed_nondx_additive.stan"
            else:
                base_file = "models/reg_horseshoe_lognormal_fixed_nondx.stan"
        else:
            base_file = "models/horseshoe_lognormal_fixed_nondx.stan"
        base_file = os.path.join(os.path.dirname(__file__), base_file)
        with open(base_file, 'r') as file:
            base_model_str = file.read()

        rate_fn_str = ""
        for i in range(R.shape[0]):
            eq_str = ""
            for j in range(R.shape[1] - 1):
                if R[i, j] == 1:
                    eq_str += "* y[{}]".format(j + 1)
                elif R[i, j] == 2:
                    eq_str += "* y[{}] * y[{}]".format(j + 1, j + 1)
            rate_fn_str += "v[{}] = theta[{}] {}; \n".format(
                i + 1, i + 1, eq_str)

        stoichiometry_str = np.array2string(S.T[:-1, :],
                                            max_line_width=np.inf,
                                            threshold=np.inf,
                                            separator=",")

        return model_str.format(R.shape[1] - 1,
                                R.shape[0],
                                R.shape[1] - 1,
                                R.shape[0],
                                stoichiometry_str,
                                rate_fn_str) + base_model_str

    def _fit_non_dx(self,
                    X0,
                    X_obs,
                    ts,
                    S,
                    R,
                    observed_species_indices,
                    regularized = True,
                    additive = False,
                    known_rates=[],
                    fit_params={},
                    model_params={}):

        data = {
            'N': ts[1:].shape[0],
            'M': S.shape[1] - 1,
            'M_obs' : len(observed_species_indices),
            'obs_idx' : [i + 1 for i in observed_species_indices], # stan idx
            'D': S.shape[0],
            'D1': len(known_rates),
            'y0' : X0,
            'y': X_obs,
            'ts': ts,
            'known_rates': known_rates
        }

        default_model_params = {'m0': 10,
                                'slab_scale': 1,
                                'slab_df': 2,
                                'tau0': 0.001,
                                'noise_sigma': 1}
        default_model_params = {**default_model_params, **model_params}
        data = {**data, **default_model_params}

        default_fit_params = {'chains': 4,
                              'iter_warmup': 1000,
                              'iter_sampling': 1000,
                              'optimize': False,
                              'init': None,
                              'show_progress': False,
                              'max_treedepth': 10,
                              'variational': False,
                              'algorithm' : 'meanfield',
                              'v_iters': 1000,
                              'v_grad_samples': None,
                              'v_elbo_samples': None,
                              'prior_predictive' : False}
        default_fit_params = {**default_fit_params, **fit_params}

        model_str = self._create_non_derivative_stan_model(
            S, R,
            regularized, additive, default_fit_params['prior_predictive']
        )
        fp = tempfile.NamedTemporaryFile(mode='w+t', suffix=".stan")
        fp.write(model_str)
        fp.seek(0)

        model = CmdStanModel(stan_file=fp.name)

        if default_fit_params['optimize']:
            fit = model.optimize(data=data,
                                 inits=default_fit_params['init'])
        elif default_fit_params['variational']:
            fit = model.variational(
                data=data,
                iter=default_fit_params['v_iters'],
                grad_samples=default_fit_params['v_grad_samples'],
                elbo_samples=default_fit_params['v_elbo_samples'],
                algorithm=default_fit_params['algorithm'],
                require_converged=False)
            fit.variational_sample.columns = fit.column_names
        else:
            if  default_fit_params['prior_predictive']:
                fit = model.sample(
                    data=data,
                    iter_sampling=default_fit_params['iter_sampling'],
                    inits=default_fit_params['init'],
                    show_progress=default_fit_params['show_progress'],
                    refresh=1,
                    fixed_param=True)
            else:
                fit = model.sample(
                    data=data,
                    chains=default_fit_params['chains'],
                    iter_warmup=default_fit_params['iter_warmup'],
                    iter_sampling=default_fit_params['iter_sampling'],
                    inits=default_fit_params['init'],
                    refresh=1,
                    show_progress=default_fit_params['show_progress'],
                    max_treedepth=default_fit_params['max_treedepth'])
        fp.close()
        return fit
