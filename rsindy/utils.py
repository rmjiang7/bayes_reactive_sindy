import numpy as np
import itertools
from scipy import integrate

def prune_reactions(fit, threshold = 1e-3, pthresh = 0.9):
    tau = fit.stan_variables()['tau']
    lambda_tilde = fit.stan_variables()['lambda_tilde']
    probs = np.sum((lambda_tilde * tau.reshape(-1,1) > threshold), axis = 0)/tau.shape[0]
    return probs, np.where(probs > pthresh)[0]

def solve_estimated_dynamics(rates, S, R, y0, t, thresh = 0):
    est_rates = rates.copy()
    est_rates[est_rates < thresh] = 0
    def dZdt_inf(Z, t = 0):
        ap = np.hstack([Z, 1]) * (R == 1)
        ap += (np.hstack([Z, 1]) * (R == 2)) ** 2
        ap_mask = ap + (ap == 0).astype(np.float32)
        extend_Z = np.prod(ap_mask, axis = 1) * est_rates
        return (S.T @ extend_Z)[:-1]
    Z_obs_inf = integrate.odeint(dZdt_inf, y0, t)
    return Z_obs_inf

def convert_to_rsindy(descriptions, species):
    reactions = []
    species_to_idx_map = {s: i for i, s in enumerate(species)}
    for desc in descriptions:
        reactants, products = desc.split("->")
        if "+" in reactants:
            r = list(
                map(lambda x: species_to_idx_map[x.strip()], reactants.split("+")))
            if "+" in products:
                p = list(
                    map(lambda x: species_to_idx_map[x.strip()], reactants.split("+")))
                reactions.append(
                    "result.add_double_conversion([{},{}],[{},{}])".format(r[0], r[1], p[0], p[1]))
            elif "2" in products:
                p = species_to_idx_map[products.replace("2", "").strip()]
                reactions.append(
                    "result.add_double_conversion([{},{}],[{},{}])".format(r[0], r[1], p, p))
            elif products == "0":
                reactions.append(
                    "result.add_fusion({},{},{})".format(r[0], r[1], "None"))
            else:
                reactions.append("result.add_fusion({},{},{})".format(
                    r[0], r[1], species_to_idx_map[products.strip()]))
        elif "2" in reactants:
            r = species_to_idx_map[reactants.replace("2", "").strip()]
            if "+" in products:
                p = list(
                    map(lambda x: species_to_idx_map[x.strip()], products.split("+")))
                reactions.append(
                    "result.add_double_conversion([{},{}],[{},{}])".format(r, r, p[0], p[1]))
            elif "2" in products:
                p = species_to_idx_map[products.replace("2", "").strip()]
                reactions.append(
                    "result.add_double_conversion([{},{}],[{},{}])".format(r, r, p, p))
            elif products.strip() == "0":
                reactions.append(
                    "result.add_fusion({},{},{})".format(r, r, "None"))
            else:
                reactions.append("result.add_fusion({},{},{})".format(
                    r, r, species_to_idx_map[products.strip()]))
        else:
            r = species_to_idx_map[reactants.strip(
            )] if reactants.strip() != "0" else "None"
            if "+" in products:
                p = list(
                    map(lambda x: species_to_idx_map[x.strip()], products.split("+")))
                reactions.append(
                    "result.add_fission({},{},{})".format(r, p[0], p[1]))
            elif "2" in products:
                p = species_to_idx_map[products.replace("2", "").strip()]
                reactions.append(
                    "result.add_fission({},{},{})".format(r, p, p))
            elif products.strip() == "0":
                reactions.append("result.add_decay({})".format(r))
            else:
                reactions.append("result.add_conversion({},{})".format(
                    r, species_to_idx_map[products.strip()]))

    return reactions


def encode(stoichiometry, rate):
    return ",".join(
        stoichiometry.astype(int).astype(str)
    ) + "-" + ",".join(
        rate.astype(int).astype(str)
    )


def generate_valid_reaction_basis(species):

    species_to_idx_map = {s: i for i, s in enumerate(species)}
    stoichiometry = []
    rates = []
    reaction_description = []

    # Mono-molecular
    for i, s in enumerate(species):
        creation = np.zeros(len(species) + 1)
        creation[i] = 1
        creation_rate = np.zeros(len(species) + 1)
        creation_rate[-1] = 1
        destruction = np.zeros(len(species) + 1)
        destruction[i] = -1
        #stoichiometry += [creation, destruction]
        stoichiometry += [destruction]
        destruction_rate = np.zeros(len(species) + 1)
        destruction_rate[i] = 1
        #rates += [creation_rate, destruction_rate]
        rates += [destruction_rate]
        #reaction_description += ["0 -> {}".format(s),
    #                            "{} -> 0".format(s)]
        reaction_description += ["{} -> 0".format(s)]

        for j, z in enumerate(species):
            if z != s:
                reaction = np.zeros(len(species) + 1)
                reaction[species_to_idx_map[s]] += -1
                reaction[species_to_idx_map[z]] += 1
                desc = "{} -> {}".format(s, z)
                reaction_rate = np.zeros(len(species) + 1)
                reaction_rate[species_to_idx_map[s]] += 1
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(desc)

                reaction = np.zeros(len(species) + 1)
                reaction[species_to_idx_map[s]] += 0
                reaction[species_to_idx_map[z]] += 1
                desc = "{} -> {} + {}".format(s, s, z)
                reaction_rate = np.zeros(len(species) + 1)
                reaction_rate[species_to_idx_map[s]] += 1
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(desc)

                reaction = np.zeros(len(species) + 1)
                reaction[species_to_idx_map[s]] += -2
                reaction[species_to_idx_map[z]] += 1
                desc = "2{} -> {}".format(s, z)
                reaction_rate = np.zeros(len(species) + 1)
                reaction_rate[species_to_idx_map[s]] += 2
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(desc)

                reaction = np.zeros(len(species) + 1)
                reaction[species_to_idx_map[s]] += 2
                reaction[species_to_idx_map[z]] -= 1
                desc = "{} -> 2{}".format(z, s)
                reaction_rate = np.zeros(len(species) + 1)
                reaction_rate[species_to_idx_map[z]] += 1
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(desc)

            else:
                reaction = np.zeros(len(species) + 1)
                reaction[species_to_idx_map[s]] += 1
                desc = "{} -> 2{}".format(s, s)
                reaction_rate = np.zeros(len(species) + 1)
                reaction_rate[species_to_idx_map[s]] += 1
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(desc)

        for z in itertools.combinations(species, 2):
            if s != z[0] and s != z[1]:
                reaction = np.zeros(len(species) + 1)
                reaction[species_to_idx_map[s]] += -1
                reaction[species_to_idx_map[z[0]]] += 1
                reaction[species_to_idx_map[z[1]]] += 1
                reaction_rate = np.zeros(len(species) + 1)
                reaction_rate[species_to_idx_map[s]] += 1
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(
                    "{} -> {} + {}".format(s, z[0], z[1]))

    # Bi-molecular
    for i, s in enumerate(itertools.combinations(species, 2)):
        for j, z in enumerate(species):

            reaction = np.zeros(len(species) + 1)
            reaction[species_to_idx_map[s[0]]] += -1
            reaction[species_to_idx_map[s[1]]] += -1
            if z == s[0]:
                reaction[species_to_idx_map[s[0]]] += 2
                desc = "{} + {} -> 2{}".format(s[0], s[1], s[0])
            elif z == s[1]:
                reaction[species_to_idx_map[s[1]]] += 2
                desc = "{} + {} -> 2{}".format(s[0], s[1], s[1])
            else:
                reaction[species_to_idx_map[z]] += 1
                desc = "{} + {} -> {}".format(s[0], s[1], z)
            reaction_rate = np.zeros(len(species) + 1)
            reaction_rate[species_to_idx_map[s[0]]] += 1
            reaction_rate[species_to_idx_map[s[1]]] += 1
            stoichiometry.append(reaction)
            rates.append(reaction_rate)
            reaction_description.append(desc)

        for z in itertools.combinations(species, 2):
            reaction = np.zeros(len(species) + 1)
            reaction[species_to_idx_map[s[0]]] += -1
            reaction[species_to_idx_map[s[1]]] += -1
            reaction[species_to_idx_map[z[0]]] += 1
            reaction[species_to_idx_map[z[1]]] += 1
            reaction_rate = np.zeros(len(species) + 1)
            reaction_rate[species_to_idx_map[s[0]]] += 1
            reaction_rate[species_to_idx_map[s[1]]] += 1
            if not (reaction == 0).all():
                stoichiometry.append(reaction)
                rates.append(reaction_rate)
                reaction_description.append(
                    "{} + {} -> {} + {}".format(s[0], s[1], z[0], z[1]))

    reaction_mappings = {}
    for i in range(len(stoichiometry)):
        reaction_mappings[encode(stoichiometry[i], rates[i])] = i
    return np.vstack(stoichiometry), np.vstack(rates), reaction_description, reaction_mappings
