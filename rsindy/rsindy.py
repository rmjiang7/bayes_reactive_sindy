import numpy as np
import abc
from rsindy.utils import generate_valid_reaction_basis, encode


class RSINDy(object):

    def __init__(self, species_names):

        self.species_names = species_names

        network = generate_valid_reaction_basis(self.species_names)

        self.stoichiometry = network[0]
        self.rate_matrix = network[1]
        self.description = network[2]
        self.enc_to_desc = network[3]

    def get_description(self, stoichiometry, rate):

        code = encode(stoichiometry, rate)

        return self.description[self.enc_to_desc(code)]

    def remove_reactions(self, reaction_descriptions):

        for desc in reaction_descriptions:
            for i, d in enumerate(self.description):
                if desc == d:
                    self.description.pop(i)
                    self.rate_matrix = np.delete(self.rate_matrix, i, 0)
                    self.stoichiometry = np.delete(self.stoichiometry, i, 0)
                    desc_key = -1
                    for k, v in self.enc_to_desc.items():
                        if v > i:
                            self.enc_to_desc[k] = v - 1
                        elif v == i:
                            desc_key = k
                    del self.enc_to_desc[desc_key]
                    break

    @abc.abstractmethod
    def _fit_dx(self,
                X_obs,
                ts,
                S,
                R,
                known_rates,
                fit_params,
                model_params):
        pass

    @abc.abstractmethod
    def _fit_non_dx(self,
                    X_obs,
                    ts,
                    S,
                    R,
                    known_rates,
                    fit_params,
                    model_params):
        pass

    def _select_random_set(self,
                           known_S,
                           known_R,
                           N):

        reordered_rates = None
        if known_S is not None:
            # Place known stoichiometries at the top
            reordered_rates = []
            known_rx = []
            for i in range(known_S.shape[0]):
                idx = self.enc_to_desc[encode(known_S[i, :], known_R[i, :])]
                reordered_rates.append((i, idx))
                known_rx.append(idx)

            # Get the remaining available reactions
            available_reactions = list(set([i for i in range(
                0, self.stoichiometry.shape[0])]).difference(set(known_rx)))

            # Rearranged/Shuffled stoichiometry
            known_rx = np.hstack([known_rx, np.random.choice(
                available_reactions, size=N, replace=False)])
        else:
            # No known reactions, pick a random subset
            known_rx = np.random.choice(
                list(range(self.stoichiometry.shape[0])),
                size=N,
                replace=False)

        # Rearrange all terms
        r_stoichiometry = self.stoichiometry[known_rx, :]
        r_rate_matrix = self.rate_matrix[known_rx, :]
        r_descriptions = [self.description[d] for d in known_rx]
        reorder = None

        if reordered_rates is not None:
            reorder = sorted(reordered_rates, key=lambda x: x[1])

        return reorder, r_stoichiometry, r_rate_matrix, r_descriptions

    def fit_dx(self,
               X_obs,
               ts,
               known_S=None,
               known_R=None,
               known_rates=[],
               N=-1,
               fit_params={},
               model_params={},
               seed=None):

        if seed is not None:
            np.random.seed(seed)

        if N == -1:
            N = self.stoichiometry.shape[0] - known_S.shape[0]

        random_set = self._select_random_set(known_S, known_R, N)
        reorder, r_stoichiometry, r_rate_matrix, r_descriptions = random_set

        fit = self._fit_dx(X_obs,
                           ts,
                           r_stoichiometry,
                           r_rate_matrix,
                           known_rates,
                           fit_params=fit_params,
                           model_params=model_params)

        return fit, reorder, r_stoichiometry, r_rate_matrix, r_descriptions

    def fit_non_dx(self,
                   X_obs,
                   ts,
                   known_S=None,
                   known_R=None,
                   known_rates=[],
                   N=-1,
                   fit_params={},
                   model_params={},
                   seed=None):
        if seed is not None:
            np.random.seed(seed)

        if N == -1:
            N = self.stoichiometry.shape[0] - known_S.shape[0]

        random_set = self._select_random_set(known_S, known_R, N)
        reorder, r_stoichiometry, r_rate_matrix, r_descriptions = random_set

        fit = self._fit_non_dx(X_obs,
                               ts,
                               r_stoichiometry,
                               r_rate_matrix,
                               known_rates,
                               fit_params,
                               model_params)

        return fit, reorder, r_stoichiometry, r_rate_matrix, r_descriptions
